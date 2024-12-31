from types import NoneType, TracebackType
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    cast,
)

import abc
import atexit
import contextlib
import difflib
import functools
import hashlib
import html
import inspect
import itertools
import json
import os
import pathlib
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback
import urllib.parse
import uuid

try:
    # grequests следует импортировать перед requests: https://github.com/spyoungtech/grequests/issues/103
    import grequests
except RuntimeError:
    # Poetry, установленный nix'ом, не может найти libstdc++:
    # $ nix-env -iA nixpkgs.poetry
    # $ poetry install
    # $ poetry run python3 music.py
    # ...
    #     from ._greenlet import _C_API # pylint:disable=no-name-in-module
    #     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # ImportError: libstdc++.so.6: cannot open shared object file: No such file or directory

    import ctypes

    ctypes.cdll.LoadLibrary("libstdc++.so.6")

    import grequests  # type: ignore[import]

import bs4
import colorama
import ffpb  # type: ignore[import]
import mutagen.id3
import mutagen.mp3
import pytubefix  # type: ignore[import]
import pytubefix.exceptions  # type: ignore[import]
import pytubefix.extract  # type: ignore[import]
import pytubefix.request  # type: ignore[import]
import pyyoutube  # type: ignore[import]
import requests
import tqdm
import tqdm.utils
import vk_api  # type: ignore[import]
import vk_api.audio  # type: ignore[import]
import yandex_music
import ytmusicapi  # type: ignore[import]


# mypy понимает условия, содержащие `sys.platform`, что позволяет писать платформоспецифичный код,
# но теряется, как только проверки ОС помещаются в переменную:
#
#   OS_WINDOWS = sys.platform == "win32"
#   reveal_type(OS_WINDOWS)     ■ Revealed type is "builtins.bool"
#   if OS_WINDOWS:
#       import msvcrt
#       return msvcrt.getch().decode()      ■ Module has no attribute "getch"
#
# (см. https://mypy.readthedocs.io/en/stable/common_issues.html#python-version-and-system-platform-checks)
#
# К счастью, использование переменных с литеральным типом решает эту проблему
if sys.platform == "darwin":
    OS_DARWIN: Literal[True] = True
    OS_WINDOWS: Literal[False] = False
    OS_LINUX: Literal[False] = False
elif sys.platform == "linux":
    OS_DARWIN: Literal[False] = False
    OS_WINDOWS: Literal[False] = False
    OS_LINUX: Literal[True] = True
elif sys.platform == "win32":
    OS_DARWIN: Literal[False] = False
    OS_WINDOWS: Literal[True] = True
    OS_LINUX: Literal[False] = False
# `OS_ANDROID` имеет нелитеральный тип `builtins.bool`
OS_ANDROID = hasattr(sys, "getandroidapilevel")


if OS_WINDOWS:
    import msvcrt
else:
    import termios
    import tty


#
# Конфигурация
#


def ensure_ffmpeg_installed() -> None:
    process = subprocess.run("ffmpeg -version", capture_output=True, shell=True, text=True)
    if process.returncode == 127:
        exit("ffmpeg: requirement is not satisfied")


#
# Патчи
#


def patch_tqdm() -> None:
    """
    На Windows tqdm некорректно определяет ширину бара при `ncols=None`: возвращается значение,
    меньшее ожидаемого на 1, из-за чего

        for it in tqdm.tqdm(...):
            if ...:
                print("Go bananas")

    выводит

        Buzzing on Fuzzing:  70%|██▊ | 7/10 [00:18<00:08,  2.67s/it]G
        o bananas

    В то время как на Linux

        Buzzing on Fuzzing:  70%|██▊ | 7/10 [00:18<00:08,  2.67s/it]
        Go bananas

    tqdm вычисляет ширину терминала как разницу координат правой и левой сторон:
        https://github.com/tqdm/tqdm/blob/6791e8c5b3d6c30bdd2060c346996bfb5a6f10d1/tqdm/utils.py#L264
        https://learn.microsoft.com/en-us/windows/console/console-screen-buffer-info-str, см. srWindow

    Для сравнения, в CPython ширина определяется как разница координат плюс 1:
        https://github.com/python/cpython/blob/5a2b984568f72f0d7ff7c7b4ee8ce31af9fd1b7e/Modules/posixmodule.c#L13473
    """
    monkey_patch(
        tqdm.utils._screen_shape_windows,  # type: ignore[attr-defined]
        "585e19a361bbf8f3fbe1ba7fffd7ccb6a2baab8a697e28d1ca28af26",
        """\
@@ -16,7 +16,7 @@
         if res:
             (_bufx, _bufy, _curx, _cury, _wattr, left, top, right, bottom,
              _maxx, _maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
-            return right - left, bottom - top  # +1
+            return right - left + 1, bottom - top + 1
     except Exception:  # nosec
         pass
     return None, None
""",
    )


def patch_vk_api() -> None:
    # Примерно с 06.06.2023 vk_api начал падать с ошибкой `vk_api.exceptions.AuthError: Unknown API auth error`
    # Может быть, это связано с недавними падениями - не знаю
    monkey_patch(
        vk_api.VkApi._api_login,
        "6b67e4eaee01e627cda4d227f0956712be83e552b83415d76f9f4a94",
        """\
@@ -16,6 +16,9 @@
         }
     )

+    if 'redirect_uri' in response.url:
+        response.url = input(f'Enter authorization data {response.url}: ')
+
     if 'act=blocked' in response.url:
         raise AccountBlocked('Account is blocked')
""",
    )


def monkey_patch(f: Callable[..., Any], old_source_digest: str, patch: str) -> None:
    """Применяет unified-патч к исходному коду"""
    source = inspect.getsource(f)
    source_digest = hashlib.sha224(source.encode()).hexdigest()
    assert old_source_digest == source_digest, f"`{f.__name__}` source has been modified"
    modified_source = apply(textwrap.dedent(source), patch)
    modified_code = compile(modified_source, "<string>", "exec")
    module: dict[str, Any] = {}
    exec(modified_code, module)
    f.__code__ = module[f.__name__].__code__


#
# Типы
#


T = TypeVar("T")


#
# Типажи
#


class Show(abc.ABC):
    @abc.abstractmethod
    def show(self) -> str:
        pass

    def __repr__(self) -> str:
        return self.show()

    def __str__(self) -> str:
        return self.show()


#
# Коллекции
#


class AutovivificiousDict(dict[str, Any]):
    """
    'Автоматически оживляемый словарь' - https://en.wikipedia.org/wiki/Autovivification

    Позволяет ссылаться на произвольные значения словаря без их явного объявления:

    >>> d = AutovivificiousDict()
    >>> d['foo']['bar']['baz'] = 42
    >>> d
    {'foo': {'bar': {'baz': 42}}}
    """

    def __missing__(self, key: str) -> Any:
        self[key] = AutovivificiousDict()
        return self[key]


def flatten(obj: Any) -> Iterator[Any]:
    """
    'Сглаживающий' итератор для произвольно вложенной коллекции:

    >>> list(flatten([1, [[2, 3], 4, [5]]]))
    [1, 2, 3, 4, 5]
    """
    for it in obj:
        if isinstance(it, (list, tuple)):
            yield from flatten(it)
        else:
            yield it


# Диапазоны hunk'ов в unified-формате
HUNK_RANGE_RE = re.compile(r"^@@ -(\d+)(,\d+)? \+(\d+)(,\d+)? @@$")


def apply(original: str, patch: str) -> str:
    """
    Применяет unified-патч к строке

    `apply` не добавляет символ новой строки в конец строки ("\\ No new line at end of file" игнорируется)
    """
    # Почитать о структуре hunk'ов можно здесь: https://en.wikipedia.org/wiki/Diff#Unified_format
    result = original.splitlines(True)
    offset = 0
    start = 0
    for it in patch.splitlines(True):
        if it.startswith(("---", "+++")):
            pass
        elif it.startswith("@@"):
            match = re.match(HUNK_RANGE_RE, it)
            if match is None:
                raise ValueError("invalid hunk range")
            # Есть два файла:
            #
            #   $ cat original                         $ cat new
            #   Я так давно родился,                   Я так давно родился,
            #   Что слышу иногда,                      Что слышу иногда,
            #   Как надо мной проходит                 Как надо мной проходит
            #   Студеная вода.                         Студеная вода.
            #
            #                                          (с) Арсений Тарковский
            #
            # Обращаем внимание, что несмотря на разное количество контекстных строк,
            # оба изменения начинаются с 4-ой строки:
            #
            #   $ diff original new --unified=0        $ diff original new --unified=1
            #   --- original                           --- original
            #   +++ new                                +++ new
            #   @@ -4,0 +5,2 @@                        @@ -4 +4,3 @@
            #   +                                       Студеная вода.
            #   +(с) Арсений Тарковский                +
            #                                          +(с) Арсений Тарковский
            #
            # Если не прибавить единичку к `start`'у, когда количество изменённых строк в оригинальном файле
            # равно нулю (`match.group(2)`), патч будет применён неправильно
            start = int(match.group(1)) - 1 + (1 if match.group(2) == ",0" else 0)
        # Думаю, что подавляющее количество патчей будут заданы многострочными литералами строки в исходном коде
        # Значимые конечные пробелы патча могут быть случайно удалены командой Trim Trailing Whitespace редактора
        # А если и не будут удалены, то вызовут предупреждение flake'а "W293 blank line contains whitespace"
        # Поэтому воспринимаем переход на новую строку как пустую контекстную строку (' \n')
        elif it.startswith((" ", "\n")):
            if result[start + offset] != it.removeprefix(" "):
                raise ValueError(
                    "invalid hunk: line {}: got {!r}, expected {!r}".format(
                        start + 1,
                        result[start + offset],
                        it.removeprefix(" "),
                    )
                )
            start += 1
        elif it.startswith("-"):
            if result[start + offset] != it.removeprefix("-"):
                raise ValueError(
                    "invalid hunk: line {}: got {!r}, expected {!r}".format(
                        start + 1,
                        result[start + offset],
                        it.removeprefix("-"),
                    )
                )
            del result[start + offset]
            offset -= 1
            start += 1
        elif it.startswith("+"):
            result.insert(start + offset, it.removeprefix("+"))
            offset += 1
    return "".join(result)


#
# Ввод/вывод
#


def split_ansi(s: str) -> Iterator[str]:
    """
    Разбивает строку на символы и управляющие ANSI-последовательности:

    >>> list(split_ansi('\x1b[31mfoo\x1b[0m'))
    ['\\x1b[31m', 'f', 'o', 'o', '\\x1b[0m']
    """
    for i, it in enumerate(re.split(r"(\x1b\[\d+m)", s)):
        if i % 2 == 0:
            yield from it
        else:
            yield it


def visual_length(s: str) -> int:
    """
    Длина строки без учёта управляющих ANSI-последовательностей:

    >>> visual_length('\x1b[31mfoo\x1b[0m')
    3
    """
    return sum(it.isprintable() for it in split_ansi(s))


def fit(s: str, width: int, placeholder: str = "...") -> str:
    """
    Подрезает строку, чтобы она вписывалась в заданную ширину:

    >>> fit("Hello, World!", 10)
    'Hello, ...'
    >>> fit("Hello, World!", 10, placeholder='>')
    'Hello, Wo>'
    >>> fit("Hello, World!", 20)
    'Hello, World!'
    """
    words = list(split_ansi(s))
    length = visual_length(s)

    if length > width:
        n_popped = 0
        while n_popped < length - width + visual_length(placeholder):
            n_popped += len(words[-1]) == 1
            words.pop()
        words += placeholder

    return "".join(words)


def markup(s: str) -> str:
    """Стилизация текста при помощи Markdown-разметки"""
    s = re.sub("`(.*?)`", rf"{colorama.Fore.LIGHTYELLOW_EX}\1{colorama.Style.RESET_ALL}", s)
    s = re.sub(r"\*\*(.*?)\*\*", rf"{colorama.Fore.LIGHTRED_EX}\1{colorama.Style.RESET_ALL}", s)
    s = re.sub(r"\*(.*?)\*", rf"{colorama.Fore.LIGHTBLACK_EX}\1{colorama.Style.RESET_ALL}", s)
    return s


def swrite(*objs: Any, sep: str = " ") -> str:
    """Стилизованный вывод в строку"""
    output = sep.join(map(str, objs))
    output = markup(output)
    output = fit(output, os.get_terminal_size()[0], f"{colorama.Style.RESET_ALL}…")
    return output


def write(*objs: Any, end: str = "\n", flush: bool = False, sep: str = " ") -> None:
    """Стилизованный вывод"""
    print(swrite(*objs, sep=sep), end=end, flush=flush)


class Status:
    """Индикатор состояния с анимацией"""

    def __init__(self, message: str):
        self.__message = message
        self.__postfix = ""
        self._is_failed = False

    def set_message(self, s: str) -> None:
        self.__message = s
        self.render()

    def set_postfix(self, s: str) -> None:
        self.__postfix = s
        self.render()

    def __enter__(self) -> "Status":
        self.__postfix = "..."

        output = swrite(f"{self.__message}{self.__postfix}")
        # См. комментарий в `render`'е
        padding = " " * (os.get_terminal_size()[0] - visual_length(output))
        # Не делаем возврата каретки, чтобы не перетереть внешний `Status`
        print(f"{output}{padding}", end="", flush=True)

        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if not self._is_failed:
            self.set_postfix(", done" if exc_type is None else ", failed")
            write()
        else:
            # Сообщение об ошибке было выведено `fail`'ом - ничего не делаем
            pass

    def fail(self, cause: str = "") -> None:
        self._is_failed = True
        self.set_postfix(", failed" + (cause and f": {cause}" or ""))

    def render(self) -> None:
        output = swrite(f"{self.__message}{self.__postfix}")
        # Благодаря `padding`'у вложенные `Status`'ы и `write`'ы будут выводиться на новой строчке
        # без явного вызова перевода строки
        padding = " " * (os.get_terminal_size()[0] - visual_length(output))
        print(f"\r{output}{padding}", end="", flush=True)


def read_password(prompt: str = "") -> str:
    """Ввод пароля с заменой выводимых символов на звёздочки"""

    write(prompt, flush=True, end="")

    if not OS_WINDOWS:
        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        tty.setraw(fd)

    result = ""
    try:
        while True:
            if OS_WINDOWS:
                code = msvcrt.getch()
            else:
                code = ord(sys.stdin.read(1))

            # ETX (end of text: код "03" используется для отправки процессу сигнала "SIGINT")
            if code == 3:
                raise KeyboardInterrupt()
            # BS (backspace) и FF (form feed)
            elif code in (8, 127):
                if result:
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                    result = result[:-1]
            # LF (line feed)
            elif code == 13:
                break
            # Непечатаемые символы
            elif 0 <= code <= 31:
                pass
            else:
                sys.stdout.write("*")
                sys.stdout.flush()
                result += chr(code)
    finally:
        if not OS_WINDOWS:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        sys.stdout.write("\n")
    return "".join(result)


def ffmpeg_with_progress_bar(
    *,
    ffmpeg_args: list[str],
    tqdm_ascii: bool | str | None = None,
    tqdm_desc: str | None = None,
) -> None:
    """
    Запускает ffmpeg с аргументами `ffmpeg_args` и отображает прогресс выполнения команды
    Параметры tqdm'а сильно ограничены ввиду чрезмерной хрупкости кода
    """

    bitrate = 1.0

    class Bar(tqdm.tqdm):  # type: ignore[type-arg]
        def __init__(self, **kwargs: dict[str, Any]):
            assert isinstance(kwargs["total"], int)

            super().__init__(
                ascii=tqdm_ascii,
                desc=tqdm_desc,
                total=kwargs["total"],
                # См. `Bar.format_meter`
                unit_divisor=1024,
                unit_scale=True,
                unit="B",
            )

        @staticmethod
        def format_meter(**kwargs):  # type: ignore[no-untyped-def]
            """
            `ffpb.ProgressNotifier` показывает прогресс загрузки трека в секундах:

                # Скачано 2:26 минуты из 5:29
                Receiving track:  44%|::::::::::..............| 146/329 [00:01<00:02, 82.43it/s]

            Можно было бы предположить, что для отображения прогресса скачивания в байтах достаточно
            передать в `super().__init__` параметры `unit='B'`, `unit_divisor=1024` и `unit_scale=bitrate / 8`,
            но, к сожалению, это не сработает: tqdm не добавляет приставки СИ, если `unit_scale` не равен `True`
            (см. https://github.com/tqdm/tqdm/issues/765#issuecomment-545366526):

                # В правой части ожидалось '5.70M/12.9M [00:01<00:02, 2.95MB/s]'
                Receiving track:  44%|:::....| 5980160.0/13475840.0 [00:01<00:02, 3098533.37B/s]

            Ввиду этого переопределяем метод `format_meter` и вручную масштабируем параметры
            """
            # Идентично https://github.com/tqdm/tqdm/blob/0bb91857eca0d4aea08f66cf1c8949abe0cd6b7a/tqdm/std.py#L427,
            # за тем исключением, что нет строчки `unit_scale = False`
            if kwargs["total"]:
                kwargs["total"] *= bitrate / 8
            kwargs["n"] *= bitrate / 8
            if kwargs["rate"]:
                kwargs["rate"] *= bitrate / 8
            return tqdm.tqdm.format_meter(**kwargs)

        def update(self, n: float | None = 1) -> bool | None:
            """
            Длительность трека "Черная дыра (SLOWED BY Ĉarrier Ħeroin) - Kunteynir"
            (https://vk.com/audio240489972_456241117_82c83447669f2d5518) составляет 7:55 минут,
            но ffpb выставляет `self.total` в 474 секунды (7:54 минуты):

                # Предпоследняя итерация
                Receiving track:  92%|:::::::::::::::::::..| 17.0M/18.5M [00:03<00:00, 4.74MB/s]

                # Последняя итерация: исчез прогресс выполнения, и размер файла изменился с 18.5MB на 18.6MB
                Receiving track: 18.6MB [00:04, 4.71MB/s]
            """
            self.total = max(self.total, self.n + n)
            return super().update(n)

    with ffpb.ProgressNotifier(tqdm=Bar) as notifier:
        process = subprocess.Popen(["ffmpeg"] + ffmpeg_args, stderr=subprocess.PIPE)

        while True:
            if stream := process.stderr:
                if data := stream.read(1):
                    # Битрейт может меняться от фрагмента к фрагменту,
                    # поэтому ищем значение битрейта в каждой строке вывода ffmpeg'а - это актуально
                    # при скачивании треков из ВКонтакте и конвертировании mp4 в mp3
                    if data in b"\r\n":
                        if found := re.search(r"bitrate=\s*(.+?)kbits/s", notifier.line_acc.decode()):
                            bitrate = float(found.group(1)) * 1024
                    notifier(data)
                elif process.poll() is not None:
                    break


#
# Файловая система
#


if OS_ANDROID:
    MUSIC_FOLDER = pathlib.Path("/storage/emulated/0/Music")
elif OS_DARWIN:
    MUSIC_FOLDER = pathlib.Path(f"/Users/{os.getlogin()}/Music")
elif OS_LINUX:
    MUSIC_FOLDER = pathlib.Path.home() / "Downloads" / "Music"
elif OS_WINDOWS:
    MUSIC_FOLDER = pathlib.Path(os.environ["USERPROFILE"], "Music")
else:
    assert False, "Unknown platform"


def remove_invalid_path_chars(s: str) -> str:
    """Удаляет из строки символы, недопустимые в именах путей"""
    return re.sub(r'[:?"*/\<>|]', "", s)


@contextlib.contextmanager
def atomic_path(path: pathlib.Path, suffix: str = "") -> Iterator[pathlib.Path]:
    """
    Гарантирует отсутствие файлов, находящихся в промежуточном состоянии:
    при успешном выполнении контекста файл перемещается из /tmp в `path`
    """
    tmp_path = pathlib.Path(tempfile.gettempdir()) / f"{uuid.uuid4()}{suffix}"

    try:
        yield tmp_path
    except (Exception, KeyboardInterrupt):
        raise
    else:
        # На Android'е `tmp_path.rename(path)` выбросит исключение `OSError: [Errno 18] Invalid cross-device link`,
        # из-за разных файловых систем Termux'а и `path`'а: /data и /storage
        # (см. https://stackoverflow.com/questions/42392600/oserror-errno-18-invalid-cross-device-link)
        shutil.move(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


#
# Сеть
#


class ProxiedRequests:
    """Выполнение запросов из-под прокси-сервера"""

    def __init__(self) -> None:
        # Всё же поиск рабочих прокси-серверов - долгая операция,
        # так что запоминаем их адреса и используем повторно
        self._netloc_proxies: dict[str, str] = {}

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        stream: bool | None = None,
        timeout: float | None = None,
    ) -> requests.Response:
        return self.request("get", url, params=params, stream=stream, timeout=timeout)

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        stream: bool | None = None,
        timeout: float | None = None,
    ) -> requests.Response:
        netloc = urllib.parse.urlparse(url)[1]
        if netloc in self._netloc_proxies:
            try:
                proxy = self._netloc_proxies[netloc]
                return requests.request(
                    method,
                    url,
                    params=params,
                    proxies={"http": proxy, "https": proxy},
                    stream=stream,
                    timeout=timeout,
                )
            except Exception:
                pass

        # Первым порывом во время рефакторинга будет удаление цикла и
        # превращение `requests_` в бесконечный генератор `(grequests.request(...) for proxy in self._proxies())`
        # Такой код приведёт к проблемам с SSL-сертификатами: после выхода из функции greenlet-потоки,
        # лежащие в основе grequests, продолжат делать запросы несмотря на то, что ассоциирующийся с ними генератор умер
        proxies = self._proxies()
        while True:
            # Количество одновременно выполняемых запросов
            # Значение 128 взято с потолка - оно не должно быть слишком маленьким, чтобы скорость метода
            # была на хорошем уровне, но и не должно быть слишком большим, чтобы не попасть в ловушку,
            # описанную комментарием выше
            size = 128
            requests_ = (
                grequests.request(
                    method,
                    url,
                    # `grequests.imap` возвращает ответы в неупорядоченном виде,
                    # поэтому добавляем адрес прокси-сервера в заголовки,
                    # чтобы позже иметь возможность восстановить его
                    headers={"proxy": proxy},
                    params=params,
                    proxies={"http": proxy, "https": proxy},
                    stream=stream,
                    timeout=timeout,
                )
                for proxy in itertools.islice(proxies, size)
            )
            responses: Iterator[requests.Response] = grequests.imap(requests_, size=None)
            for it in responses:
                if it:
                    self._netloc_proxies[netloc] = it.request.headers["proxy"]
                    return it

    def _proxies(self) -> Iterator[str]:
        """
        Бесконечный генератор, возвращающий адреса прокси-серверов
        Адреса могут повторяться
        """

        while True:
            try:
                response = requests.get("https://www.sslproxies.org").text
                yield from re.findall(r"\d+\.\d+\.\d+\.\d+:\d+", response)
            except requests.RequestException:
                pass


proxied_requests = ProxiedRequests()


#
# Кэширование
#


@functools.cache  # Для идентичности объектов кэша при повторном вызове
def cache_file(path: pathlib.Path) -> AutovivificiousDict:
    """
    Кэш-файл, представляемый автоматически оживляемым словарём (см. `AutovivificiousDict`)
    В течение работы программы данные хранятся в ОЗУ: выгрузка осуществляется при завершении программы
    """

    def load() -> AutovivificiousDict:
        with path.open("rb") as file:
            try:
                return pickle.load(file)  # type: ignore[no-any-return]
            except EOFError:
                return AutovivificiousDict()

    def dump(cache: AutovivificiousDict) -> None:
        with path.open("wb") as file:
            pickle.dump(cache, file)

    path.touch(exist_ok=True)
    cache = load()
    atexit.register(lambda: dump(cache))

    return cache


#
# База данных
#


class Database:
    def __init__(self) -> None:
        self._bandcamp_database = BandcampDatabase()
        self._genius_database = GeniusDatabase()
        self._youtube_music_database = YouTubeMusicDatabase()

    def search_track(self, query: str) -> "DatabaseTrack":
        return DatabaseTrack(self, Database._optimized_query(query))

    @staticmethod
    def _optimized_query(query: str) -> str:
        """
        Оптимизирует запрос для поиска, удаляя из него лишние фрагменты:

        >>> Database._optimized_query('Хаски - Ай [Slowed + Reverb] By THAWANTEDZ ⑦⑦⑦')
        'Хаски - Ай'

        Иногда метод ведёт себя слишком агрессивно, удаляя существенные части из строки:

        >>> Database._optimized_query('Lana Del Rey - High By The Beach')
        'Lana Del Rey - High'

        но в целом бэкенд-сервисы прекрасно справляются с такими запросами
        """

        def strip_author(s: str) -> str:
            s = re.sub(r"(.*)\sby\s.*", r"\1", s, flags=re.IGNORECASE)
            return s

        query = re.sub(r"\s+", " ", query)
        parens_stripped = re.sub(r"\s*[\{\[\(].*?[\)\]\}]", "", query)
        # Оставляем скобки, если исполнитель или название песни целиком в них заключены
        if len(parens_stripped.split(" - ")) >= 2:
            query = parens_stripped
        query = " - ".join(filter(None, map(strip_author, query.split(" - "))))
        return query


class DatabaseTrack:
    def __init__(self, database: Database, query: str):
        # Доверие к результатам выдачи базы данных
        database_trust = {
            YouTubeMusicDatabaseTrack: 3,
            BandcampDatabaseTrack: 2,
            GeniusDatabaseTrack: 1,
            NoneType: 0,
        }

        self._database = database
        self._query = query

        self._matches = [
            self._database._bandcamp_database.search_track(self._query),
            self._database._genius_database.search_track(self._query),
            self._database._youtube_music_database.search_track(self._query),
        ]
        self._matches.sort(
            # Возвращаем кортеж, чтобы выдачи сравнивались по `database_trust`
            # в случае одинаковых коэффициентов `_query_ratio`
            key=lambda x: (
                DatabaseTrack._query_ratio(query, f"{x.artists} - {x.title}".lower()) if x is not None else 0,
                database_trust[type(x)],
            ),
            reverse=True,
        )

    @staticmethod
    def _query_ratio(query: str, found: str) -> float:
        """
        Возвращает меру сходства запроса и найденного трека

        Игнорирует порядок, в котором указаны исполнители и название песни, чтобы не получать каверы там,
        где есть возможность получить оригинальный трек:

        >>> def dummy_query_ratio(query, found):
        ...     '''Реализация `DatabaseTrack._query_ratio` без игнорирования порядка'''
        ...     return difflib.SequenceMatcher(a=query, b=found).ratio()
        >>>
        >>> query = 'Idioteque - Radiohead'
        >>> bandcamp_found = 'Vlantis - Idioteque (Radiohead)'
        >>> youtube_music_found = 'Radiohead - Idioteque'
        >>>
        >>> # Сравнение "глупым" `_query_ratio`:
        >>> # `bandcamp_found` вернул большее число, поэтому метаданные будут браться с кавера
        >>> dummy_query_ratio(query, bandcamp_found)
        0.7307692307692307
        >>> dummy_query_ratio(query, youtube_music_found)
        0.42857142857142855
        >>>
        >>> # Сравнение "умным" `_query_ratio`:
        >>> # `youtube_music_found` вернул большее число, так что метаданные будут браться с оригинального трека
        >>> DatabaseTrack._query_ratio(query, bandcamp_found)
        0.425
        >>> DatabaseTrack._query_ratio(query, youtube_music_found)
        1.0
        """

        query = query.lower()
        found = found.lower()

        query_parts = query.split(" - ")
        found_parts = found.split(" - ")

        if len(query_parts) == 1 or len(found_parts) == 1:
            return difflib.SequenceMatcher(a=query, b=found).ratio()

        artists_artists_ratio = difflib.SequenceMatcher(a=query_parts[0], b=found_parts[0]).ratio()
        title_title_ratio = difflib.SequenceMatcher(a=query_parts[1], b=found_parts[1]).ratio()

        artists_title_ratio = difflib.SequenceMatcher(a=query_parts[0], b=found_parts[1]).ratio()
        title_artists_ratio = difflib.SequenceMatcher(a=query_parts[1], b=found_parts[0]).ratio()

        return max(artists_artists_ratio + title_title_ratio, artists_title_ratio + title_artists_ratio) / 2

    @property
    def album(self) -> str:
        for it in self._matches:
            if result := it is not None and it.album:
                return result
        return ""

    @property
    def artists(self) -> str:
        for it in self._matches:
            if result := it is not None and it.artists:
                return result
        return ""

    @property
    def cover(self) -> bytes:
        for it in self._matches:
            if result := it is not None and it.cover:
                return result
        return b""

    @property
    def lyrics(self) -> str:
        for it in self._matches:
            if result := it is not None and it.lyrics:
                return result
        return ""

    @property
    def title(self) -> str:
        for it in self._matches:
            if result := it is not None and it.title:
                return result
        return ""

    @property
    def year(self) -> str:
        for it in self._matches:
            if result := it is not None and it.year:
                return result
        return ""


class BandcampDatabase:
    @staticmethod
    def get(url: str, *, params: dict[str, str] = {}) -> requests.Response:
        """
        В своё время Роспотребнадзор заблокировал домен f4.bcbits.com, на котором Bandcamp хостит свои картинки
        (https://roskomsvoboda.org/post/rpn-narushil-raboru-bandcamp-and-photobucket/),
        и, может быть из-за этого, некоторые публичные сети ограничивают доступ к сайту
        """
        try:
            return requests.get(url, params=params)
        except requests.exceptions.SSLError:
            return proxied_requests.get(url, params=params, timeout=10)

    def search_track(self, query: str) -> Optional["BandcampDatabaseTrack"]:
        response = BandcampDatabase.get("https://bandcamp.com/search", params={"q": query, "item_type": "t"})
        html_ = response.text
        soup = bs4.BeautifulSoup(html_, "html.parser")

        # Лучшее совпадение не всегда находится на первом месте
        # К примеру, "Boards of Canada - Pete Standing Alone" в поисковой выдаче лишь пятый,
        # в то время как на первых четырёх позициях мэшапы
        matches: dict[str, dict[str, str]] = {}

        for it in soup.find_all("li", class_="searchresult"):
            artists = it.find("div", class_="subhead").get_text().splitlines()[-2].strip().removeprefix("by ")
            album = it.find("div", class_="subhead").get_text().splitlines()[-4].strip().removeprefix("from ")
            title = it.find("div", class_="heading").a.get_text().strip()
            url = it.find("div", class_="heading").a["href"]
            year = it.find("div", class_="released").get_text().split()[-1]

            matches[f"{artists} - {title}"] = {
                "artists": artists,
                "album": album,
                "title": title,
                "url": url,
                "year": year,
            }

        if not matches:
            return None

        if closest_matches := difflib.get_close_matches(query, matches, n=1):
            return BandcampDatabaseTrack(**matches[closest_matches[0]])

        return None


class BandcampDatabaseTrack:
    def __init__(self, artists: str, album: str, title: str, url: str, year: str):
        album = album.replace("\u200b", "").replace("\u200e", "")
        # Иногда треки дублируют исполнителя в названии альбома: https://cleanplate.bandcamp.com/track/36-day-syndrome
        if match := re.match(f"{re.escape(artists)} - ", album, re.IGNORECASE):
            album = album[len(match.group()) :]  # noqa: E203

        title = re.sub(r"\s+", " ", title)
        title = re.sub(r"\s*[\{\[\(].*?[\)\]\}]", "", title)
        # А порой и в названии трека: https://kinogroup.bandcamp.com/track/--28
        if match := re.match(f"{re.escape(artists)} - ", title, re.IGNORECASE):
            title = title[len(match.group()) :]  # noqa: E203

        self._url = url
        self.album = album
        self.artists = artists
        # Bandcamp не предоставляет текстов песен
        self.lyrics = ""
        self.title = title
        self.year = year

    @property
    def cover(self) -> bytes:
        response = BandcampDatabase.get(self._url)
        html_ = response.text
        soup = bs4.BeautifulSoup(html_, "html.parser")

        popup_image = soup.find("a", class_="popupImage")
        assert isinstance(popup_image, bs4.Tag)
        cover_url = popup_image["href"]
        assert isinstance(cover_url, str)

        return proxied_requests.get(cover_url, timeout=10).content


class GeniusDatabase:
    # У переведённых текстов песен в качестве исполнителя указывается
    # "Genius Translations" или имя из списка https://genius.com/15897404?
    TRANSLATION_ARTIST_NAMES = {
        "Genius Albanian Translations",
        "Genius Arabic Translations",
        "Genius Azerbaijani Translations",
        "Genius Brasil Translations",
        "Genius Catalan Translations",
        "Genius Chinese Translations",
        "Genius Czech Translations",
        "Genius Dutch Translations",
        "Genius English Translations",
        "Genius Farsi Translations",
        "Genius Filipino Translations",
        "Genius French Translations",
        "Genius German Translations",
        "Genius Greek Translations",
        "Genius Hebrew Translations",
        "Genius Hindi Translation",
        "Genius Hungarian Translation",
        "Genius Icelandic Translations",
        "Genius Italian Translations",
        "Genius Japanese Translations",
        "Genius Korean Translations",
        "Genius Polish Translations",
        "Genius Romanian Translations",
        "Genius Romanizations",
        "Genius Russian Translations",
        "Genius Serbian Translations",
        "Genius Slovak Translations",
        "Genius South Africa Translations",
        "Genius Spanish Translations",
        "Genius Swedish Translations",
        "Genius Thai Translations",
        "Genius Translations",
        "Genius Turkish Translations",
        "Genius Ukrainian Translation",
        "Genius Vietnamese Translations",
    }

    def search_track(self, query: str) -> Optional["GeniusDatabaseTrack"]:
        query = GeniusDatabase._optimized_query(query)

        response = requests.get("https://genius.com/api/search/song", params={"q": query})
        if not response:
            return None

        for it in response.json()["response"]["sections"][0]["hits"]:
            if it["result"]["artist_names"] not in GeniusDatabase.TRANSLATION_ARTIST_NAMES:
                return GeniusDatabaseTrack(it["result"])

        return None

    @staticmethod
    def _optimized_query(query: str) -> str:
        """
        В секции "About" к треку "No Man's Land - Падая в ночь" (https://genius.com/Nomans-land--lyrics)
        VITZYYz0r пишет: genius.com решил автоматически поменять имя авторов.

        Из-за разницы в апостроф Genius перестаёт находить трек:
            - https://genius.com/search?q=No%20Man%27s%20Land%20-%20%D0%9F%D0%B0%D0%B4%D0%B0%D1%8F%20%D0%B2%20%D0%BD%D0%BE%D1%87%D1%8C  # noqa: E501
            - https://genius.com/search?q=No%20Mans%20Land%20-%20%D0%9F%D0%B0%D0%B4%D0%B0%D1%8F%20%D0%B2%20%D0%BD%D0%BE%D1%87%D1%8C     # noqa: E501

        Поэтому удаляем апострофы из запроса, чтобы избежать подобных ошибок
        """
        return query.replace("'", "")


class GeniusDatabaseTrack:
    def __init__(self, impl: dict[str, Any]):
        self._album_url = f"https://genius.com/api/songs/{impl['id']}"
        self._cover_urls = (str(impl["song_art_image_url"]), str(impl["header_image_url"]))
        self._lyrics_url = str(impl["url"])
        self.artists = GeniusDatabaseTrack._strip_translation(impl["primary_artist"]["name"])
        self.title = GeniusDatabaseTrack._strip_translation(impl["title"])
        self.year = str(impl["release_date_components"]["year"]) if impl["release_date_components"] is not None else ""

    @staticmethod
    def _strip_translation(s: str) -> str:
        """
        Некоторые строковые данные возвращаются Genius'ом вместе с переводом на английский:

        >>> GeniusDatabaseTrack._strip_translation(
        ...     'Любимые песни (воображаемых) людей (Favorite songs of (imaginary) people)'
        ... )
        'Любимые песни (воображаемых) людей'
        """
        balance = 0
        for i, it in enumerate(reversed(s)):
            if it == "(":
                balance += 1
            elif it == ")":
                balance -= 1

            if balance == 0:
                break

        if i == 0:
            return s
        return s[: len(s) - i - 2]

    @property
    def album(self) -> str:
        response = requests.get(self._album_url)
        if album := response.json()["response"]["song"]["album"]:
            return GeniusDatabaseTrack._strip_translation(album["name"])
        return ""

    @property
    def cover(self) -> bytes:
        for it in self._cover_urls:
            # Если у трека нет обложки, Genius возвращает картинку со своим логотипом
            if it.startswith("https://assets.genius.com/images/default_cover_image.png"):
                continue
            try:
                return requests.get(it).content
            except requests.exceptions.ConnectionError:
                pass
        return b""

    @property
    def lyrics(self) -> str:
        html_ = requests.get(self._lyrics_url).text
        soup = bs4.BeautifulSoup(html_, "html.parser")

        result = ""
        for it in soup.find_all("div", {"data-lyrics-container": "true"}, recursive=True):
            result += GeniusDatabaseTrack._scrape_text(it) + "\n"
        result = result.strip()
        result = remove_lyrics_section_headers(result)
        return result

    @staticmethod
    def _scrape_text(element: bs4.element.PageElement) -> str:
        """Извлечение текста с учётом элементов переноса строк"""

        def impl(element: bs4.element.PageElement) -> list[str]:
            if isinstance(element, bs4.element.NavigableString):
                return [str(element)]
            elif isinstance(element, bs4.element.Tag):
                if element.name in ("inread-ad", "p", "primis-player"):
                    return [""]
                elif element.name in ("a", "b", "em", "i", "span"):
                    return [GeniusDatabaseTrack._scrape_text(it) for it in element.contents]
                elif element.name in ("br", "div"):
                    return [GeniusDatabaseTrack._scrape_text(it) for it in element.contents] + ["\n"]
                else:
                    assert False, "unreachable"
            else:
                assert False, "unreachable"

        return "".join(flatten(impl(element)))


class YouTubeMusicDatabase:
    def __init__(self) -> None:
        self._impl = ytmusicapi.YTMusic()
        # Язык возвращаемых данных задаётся параметром `language` конструктора `YTMusic`,
        # но в виду того, что библиотека не поддерживает русскую локализацию,
        # выставляем язык хоста напрямую в заголовках
        # (см. https://github.com/sigma67/ytmusicapi/tree/master/ytmusicapi/locales#readme)
        self._impl.context["context"]["client"]["hl"] = "ru"

    def search_track(self, query: str) -> Optional["YouTubeMusicDatabaseTrack"]:
        try:
            if found := self._impl.search(query, filter="songs", limit=1):
                return YouTubeMusicDatabaseTrack(self, found[0])
        # Сервер выбросил исключение >= 400
        except Exception:
            pass
        return None


class YouTubeMusicDatabaseTrack:
    def __init__(self, database: YouTubeMusicDatabase, impl: dict[str, Any]):
        self._cover_url = str(impl["thumbnails"][0]["url"]).replace("w60-h60", "w600-h600")
        self._database = database
        self._playlist = database._impl.get_watch_playlist(impl["videoId"], limit=1)
        self.album = str(impl["album"]["name"]) if impl["album"] is not None else ""
        self.artists = ", ".join(it["name"] for it in impl["artists"])
        self.title = str(impl["title"])

    @property
    def cover(self) -> bytes:
        return requests.get(self._cover_url).content

    @property
    def lyrics(self) -> str:
        try:
            result: str | None = self._database._impl.get_lyrics(self._playlist["lyrics"])["lyrics"]
            if result is None:
                return ""
            result = remove_lyrics_section_headers(result)
            return result
        except Exception:
            return ""

    @property
    def year(self) -> str:
        return str(self._playlist["tracks"][0].get("year", ""))


def remove_lyrics_section_headers(s: str) -> str:
    """Удаление '[Текст песни «…»]', '[Куплет]' и т.п. из текста песни"""
    if re.fullmatch(r"\[.*?\]", s):
        return ""
    return re.sub("\\[.*?\\]\n+", "", s)


#
# ВКонтакте
#


class VKontakteClient:
    def __init__(self, token: str, cache: AutovivificiousDict):
        session = vk_api.VkApi(
            token=token,
            captcha_handler=lambda x: x.try_again(
                input(
                    # Гиперссылки в Termux включают в себя завершающее двоеточие,
                    # ввиду чего открывается битый URL "https://api.vk.com/captcha.php?sid={sid}:" (с ":" в конце)
                    "Enter symbols from the picture {}{}: ".format(
                        x.get_url(),
                        # Windows Terminal отображает \u200b (пробел нулевой ширины) пустой ячейкой
                        # (https://github.com/microsoft/terminal/issues/8667)
                        "\u200b" if OS_ANDROID else "",
                    )
                )
            ),
        )

        info = session.get_api().users.get(user_ids=None)[0]

        self._api = session.get_api()
        self._audio = vk_api.audio.VkAudio(session)
        self._cache = cache
        self._token = token
        self.id = str(info["id"])

    def group(self, id_: str | int) -> "VKontakteGroup":
        return VKontakteGroup(self, id_)

    def user(self, id_: str | int | None = None) -> "VKontakteUser":
        return VKontakteUser(self, id_)


def make_vkontakte_client(config: AutovivificiousDict, cache: AutovivificiousDict) -> VKontakteClient:
    """Создание клиента с ранее введёнными учётными данными"""

    token = None
    if "kate_mobile_token" in config["vkontakte"]:
        token = config["vkontakte"]["kate_mobile_token"]

    while True:
        if token is None:
            token = read_password("Kate Mobile Access Token: ")

        with Status("Logging in to VKontakte") as status:
            try:
                return VKontakteClient(token, cache)
            except vk_api.exceptions.ApiError:
                status.fail("invalid access token")
            finally:
                config["vkontakte"]["kate_mobile_token"] = token
                token = None


class VKontakteGroup(Show):
    def __init__(self, client: VKontakteClient, id_: str | int):
        info = client._api.groups.get_by_id(group_id=id_)[0]

        self._client = client
        self.id = str(info["id"])
        self.name = str(info["name"])

    def show(self) -> str:
        return self.name

    @property
    def wall(self) -> "VKontakteWall":
        return VKontakteWall(self._client, f"-{self.id}")


class VKontakteUser(Show):
    def __init__(self, client: VKontakteClient, id_: str | int | None):
        info = client._api.users.get(user_ids=id_)[0]

        self._client = client
        self.full_name = f"{info['first_name']} {info['last_name']}"
        self.id = str(info["id"])

    def show(self) -> str:
        return self.full_name

    @property
    def tracks(self) -> list["VKontakteTrack"]:
        # Заголовки и параметры взяты отсюда: https://github.com/issamansur/vkpymusic

        kate_mobile_user_agent = (
            "KateMobileAndroid/56 lite-460 (Android 4.4.2; SDK 19; x86; unknown Android SDK built for x86; en)"
        )

        response = requests.post(
            "https://api.vk.com/method/audio.get",
            headers={
                "User-Agent": kate_mobile_user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "*/*",
                "Connection": "keep-alive",
            },
            data=[
                ("access_token", self._client._token),
                ("extended", 1),
                ("https", 1),
                ("lang", "ru"),
                ("owner_id", self.id),
                ("v", "5.131"),
            ],
        ).json()

        return [VKontakteTrack(it) for it in response["response"]["items"]]

    @property
    def wall(self) -> "VKontakteWall":
        return VKontakteWall(self._client, self.id)


class VKontakteWall:
    def __init__(self, client: VKontakteClient, owner_id: str):
        self._client = client
        self.owner_id = owner_id

    @property
    def posts(self) -> list["VKontaktePost"]:
        result: list[VKontaktePost] = []
        n_posts = self._client._api.wall.get(owner_id=self.owner_id)["count"]

        with tqdm.tqdm(
            ascii=".:",
            desc="Fetching posts",
            total=n_posts,
        ) as bar:
            for i in range(0, n_posts, 100):
                for it in self._client._api.wall.get(owner_id=self.owner_id, count=100, offset=i)["items"]:
                    result.append(VKontaktePost(self._client, it))
                bar.update(bar.n + 100 <= bar.total and 100 or bar.total - bar.n)

        return result


class VKontaktePost(Show):
    def __init__(self, client: VKontakteClient, impl: dict[str, Any]):
        self._client = client
        self.id = str(impl["id"])
        self.owner_id = str(impl["owner_id"])

    def show(self) -> str:
        return f"https://vk.com/wall{self.owner_id}_{self.id}"

    @property
    def tracks(self) -> list["VKontakteAttachedTrack"]:
        if self.id in self._client._cache["post_tracks"][self.owner_id]:
            return self._client._cache["post_tracks"][self.owner_id][self.id]  # type: ignore[no-any-return]

        result = []
        response = self._client._audio._vk.http.get(f"https://m.vk.com/wall{self.owner_id}_{self.id}")
        html_ = response.text
        soup = bs4.BeautifulSoup(html_, "html.parser")

        for it in soup.select("button[data-audio]"):
            assert isinstance(it["data-audio"], str)

            impl = json.loads(it["data-audio"])
            for key, value in impl.items():
                impl[key] = html.unescape(str(value))

            result.append(VKontakteAttachedTrack(impl))

        self._client._cache["post_tracks"][self.owner_id][self.id] = result
        return result


class VKontakteTrack(Show):
    def __init__(self, impl: dict[str, Any]):
        self._url = impl["url"]
        self.artists = impl["artist"]
        self.id = f"{impl['owner_id']}{impl['id']}"
        # API-метод 'audio.get' не возвращает ссылки на обложки
        self.cover = b""
        self.title = impl["title"]

    def show(self) -> str:
        return f"{self.artists} - {self.title}"

    def saturated_metadata(self, suggestion: DatabaseTrack) -> "TrackMetadata":
        return TrackMetadata(
            album=suggestion.album,
            artists=self.artists,
            # ВКонтакте возвращает обложки низкого качества (160x160 пикселей),
            # поэтому отдаём предпочтение обложке из базы данных
            cover=suggestion.cover or self.cover,
            lyrics=suggestion.lyrics,
            title=self.title,
            year=suggestion.year,
        )

    def download(self, path: pathlib.Path) -> None:
        if not self._url:
            raise TrackNotAvailable("not available")

        response = requests.get(self._url, stream=True)
        total = int(response.headers.get("content-length", 0))

        with tqdm.tqdm(
            ascii=".:",
            desc="Receiving track",
            total=total,
            unit="B",
            unit_divisor=1024,
            unit_scale=True,
        ) as bar:
            with path.open("wb") as file:
                for data in response.iter_content(1024):
                    bar.update(len(data))
                    file.write(data)


class VKontakteAttachedTrack(VKontakteTrack):
    def __init__(self, impl: dict[str, Any]):
        self._cover_url = impl["coverUrl"] != "None" and cast(str, impl["coverUrl"]) or ""
        self._url = str(impl["url"])
        self.artists = str(impl["artist"])
        self.id = f"{impl['owner_id']}{impl['id']}"
        self.title = str(impl["title"])


#
# Яндекс Музыка
#


# Не показывать сообщение с лицензией
yandex_music.Client._Client__notice_displayed = True  # type: ignore[attr-defined]


class YandexMusicClient:
    def __init__(self, token: str):
        self._impl: yandex_music.Client = yandex_music.Client(token).init()  # type: ignore[no-untyped-call]

    def user(self, id_: str | None = None) -> "YandexMusicUser":
        return YandexMusicUser(self, id_)


def make_yandex_music_client(config: AutovivificiousDict) -> YandexMusicClient:
    """Создание клиента с ранее введёнными учётными данными"""
    if "token" in config["yandex_music"]:
        token = config["yandex_music"]["token"]
    else:
        token = read_password("Yandex Music token: ")

    while True:
        with Status("Logging in to Yandex Music") as status:
            config["yandex_music"]["token"] = token
            try:
                # С пустым токеном yandex_music делает запросы из-под неавторизованного
                # пользователя, что накладывает ограничения на доступ плейлистам
                if token.isspace():
                    raise yandex_music.exceptions.UnauthorizedError()
                return YandexMusicClient(token)
            except yandex_music.exceptions.UnauthorizedError:
                status.fail("invalid token")
                token = read_password("Yandex Music token: ")


class YandexMusicUser(Show):
    def __init__(self, client: YandexMusicClient, id_: str | None):
        info_url = f"{client._impl.base_url}/users/{id_}"
        info = client._impl._request.get(info_url)
        assert isinstance(info, dict)

        self._client = client
        self.id: str = info["uid"]
        self.name: str = info["name"]

    def show(self) -> str:
        return self.name

    @property
    def tracks(self) -> list["YandexMusicTrack"]:
        if liked := self._client._impl.users_likes_tracks(self.id):
            return [YandexMusicTrack(it) for it in liked.fetch_tracks()]
        return []


class YandexMusicTrack(Show):
    def __init__(self, impl: yandex_music.Track):
        self._impl = impl

    def show(self) -> str:
        return f"{self.artists} - {self.title}"

    @property
    def album(self) -> str:
        if albums := self._impl.albums:
            return albums[0].title or ""
        return ""

    @property
    def artists(self) -> str:
        return ", ".join(self._impl.artists_name())

    @property
    def cover(self) -> bytes:
        if uri := self._impl.cover_uri:
            url = f"https://{uri.replace('%%', '600x600')}"
            return requests.get(url).content
        return b""

    @property
    def id(self) -> str:
        return str(self._impl.id)

    @property
    def lyrics(self) -> str:
        if supplement := self._impl.get_supplement():
            if lyrics := supplement.lyrics:
                return lyrics.full_lyrics
        return ""

    @property
    def title(self) -> str:
        return self._impl.title or ""

    @property
    def year(self) -> str:
        if albums := self._impl.albums:
            return str(albums[0].year) or ""
        return ""

    def saturated_metadata(self, suggestion: DatabaseTrack) -> "TrackMetadata":
        return TrackMetadata(
            album=self.album,
            artists=self.artists,
            cover=self.cover,
            lyrics=self.lyrics or suggestion.lyrics,
            title=self.title,
            year=self.year,
        )

    def download(self, path: pathlib.Path) -> None:
        try:
            url = self._impl.get_download_info()[0].get_direct_link()
            response = requests.get(url, stream=True)
            length = int(response.headers["content-length"])

            with tqdm.tqdm(
                ascii=".:",
                desc="Receiving track",
                total=length,
                unit_divisor=1024,
                unit_scale=True,
                unit="B",
            ) as bar:
                with path.open("wb") as file:
                    for data in response.iter_content(chunk_size=1024):
                        bar.update(file.write(data))
        except yandex_music.exceptions.UnauthorizedError as e:
            raise TrackNotAvailable("not available") from e


#
# YouTube
#


# `pytubefix.Stream.download` вызывает `on_progress` по загрузке каждых `default_range_size` байт:
# со значением по умолчанию (9 МБ) индикатор загрузки трека мгновенно заполняется, не предоставляя
# никакой информации о скорости загрузки трека
pytubefix.request.default_range_size = 1024**2


class YouTubeClient:
    def __init__(self, api_key: str):
        self._impl = pyyoutube.Api(api_key=api_key)
        # pyyoutube откладывает валидацию ключа API до первого запроса
        self._impl.get_channel_info(channel_id="UCK8sQmJBp8GCxrOtXWBpyEA")

    def user(self, id_: str) -> "YouTubeUser":
        return YouTubeUser(self, id_)


def make_youtube_client(config: AutovivificiousDict) -> YouTubeClient:
    """Создание клиента с ранее введёнными учётными данными"""
    if "api_key" in config["youtube"]:
        api_key = config["youtube"]["api_key"]
    else:
        api_key = read_password("YouTube API key: ")

    while True:
        with Status("Logging in to YouTube") as status:
            config["youtube"]["api_key"] = api_key
            try:
                return YouTubeClient(api_key)
            except pyyoutube.error.PyYouTubeException:
                status.fail("invalid API key")
                api_key = read_password("YouTube API key: ")


class YouTubeUser(Show):
    def __init__(self, client: YouTubeClient, id_: str):
        # `extract.channel_name` не умеет работать с новыми "@"-ссылками (https://github.com/pytube/pytube/issues/1443)
        save = pytubefix.extract.channel_name
        pytubefix.extract.channel_name = lambda x: x
        channel = pytubefix.Channel(id_)
        pytubefix.extract.channel_name = save

        self.client = client
        self.id: str = channel.channel_id
        self.name: str = channel.channel_name

    def show(self) -> str:
        return self.name

    @property
    def playlists(self) -> list["YouTubePlaylist"]:
        return [YouTubePlaylist(self.client, it) for it in self.client._impl.get_playlists(channel_id=self.id).items]


class YouTubePlaylist(Show):
    def __init__(self, client: YouTubeClient, impl: pyyoutube.Playlist):
        self.client = client
        self.id: str = impl.to_dict()["id"]
        self.title: str = impl.to_dict()["snippet"]["title"]
        self.url = f"https://youtube.com/playlist?list={self.id}"

    def show(self) -> str:
        return self.title

    @property
    def videos(self) -> list["YouTubeVideo"]:
        return [YouTubeVideo(it) for it in self.client._impl.get_playlist_items(playlist_id=self.id, count=None).items]


class YouTubeVideo(Show):
    def __init__(self, impl: pyyoutube.PlaylistItem):
        self.id: str = impl.to_dict()["snippet"]["resourceId"]["videoId"]
        self.title: str = impl.to_dict()["snippet"]["title"]
        self.url = f"https://www.youtube.com/watch?v={self.id}"

    def show(self) -> str:
        return self.title

    def saturated_metadata(self, suggestion: DatabaseTrack) -> "TrackMetadata":
        return TrackMetadata(
            album=suggestion.album,
            artists=suggestion.artists,
            cover=suggestion.cover,
            lyrics=suggestion.lyrics,
            title=self.title,
            year=suggestion.year,
        )

    def download(self, path: pathlib.Path) -> None:
        try:
            convert_to_mp3 = False

            with tqdm.tqdm(
                ascii=".:",
                desc="Receiving track",
                total=1,
                unit_divisor=1024,
                unit_scale=True,
                unit="B",
            ) as bar:

                def on_progress_callback(stream: pytubefix.Stream, chunk: bytes, bytes_remaining: int) -> None:
                    bar.n = min(bar.n + len(chunk), bar.total)
                    bar.refresh()

                streams = pytubefix.YouTube(
                    url=self.url,
                    on_progress_callback=on_progress_callback,
                    use_oauth=True
                ).streams

                impl = streams.filter(mime_type="audio/webm").order_by("abr").last()
                if impl is None:
                    impl = streams.filter(mime_type="audio/mp4").order_by("abr").last()
                    convert_to_mp3 = True
                bar.total = impl.filesize
                bar.refresh()
                impl.download(output_path=path.parent, filename=path.name)

            if convert_to_mp3:
                converted = path.with_name(f"{path.name}-mp3")
                ffmpeg_with_progress_bar(
                    ffmpeg_args=["-f", "mp4", "-i", str(path), "-f", "mp3", str(converted)],
                    tqdm_ascii=".:",
                    tqdm_desc="Converting track format",
                )
                path.write_bytes(converted.read_bytes())
        except pytubefix.exceptions.AgeRestrictedError as e:
            # Заменяем идентификатор видео на "this video"
            cause = "this video " + " ".join(str(e).split()[1:])
            raise TrackNotAvailable(cause) from e
        except pytubefix.exceptions.VideoPrivate as e:
            raise TrackNotAvailable("this video is private") from e


#
# Общее
#


class TrackNotAvailable(Exception):
    pass


class TrackMetadata:
    def __init__(self, *, album: str, artists: str, cover: bytes, lyrics: str, title: str, year: str):
        self.album = album
        self.artists = artists
        self.cover = cover
        self.lyrics = lyrics
        self.title = title
        self.year = year

    def embed(self, path: pathlib.Path) -> None:
        tags = mutagen.id3.ID3()  # type: ignore[no-untyped-call]
        tags["TALB"] = mutagen.id3.TALB(text=self.album)  # type: ignore[attr-defined]
        tags["TPE1"] = mutagen.id3.TPE1(text=self.artists)  # type: ignore[attr-defined]
        tags["APIC"] = mutagen.id3.APIC(data=self.cover)  # type: ignore[attr-defined]
        tags["USLT"] = mutagen.id3.USLT(text=self.lyrics)  # type: ignore[attr-defined]
        tags["TIT2"] = mutagen.id3.TIT2(text=self.title)  # type: ignore[attr-defined]
        tags["TDRC"] = mutagen.id3.TDRC(text=self.year)  # type: ignore[attr-defined]
        tags.save(path)


def sync(
    src_tracks: Sequence[VKontakteTrack | YandexMusicTrack | YouTubeVideo],
    dest_folder: pathlib.Path,
    database: Database,
) -> None:
    """Односторонняя синхронизация папки с треками"""
    track_ids = {it.id for it in src_tracks}
    track_indices = {it.id: i for i, it in enumerate(src_tracks)}
    uploaded_tracks = {"_".join(it.stem.split("_")[1:]): it for it in dest_folder.glob("*.mp3")}
    missing_tracks = [it for it in src_tracks if it.id not in uploaded_tracks]

    def remove_extraneous_tracks() -> None:
        for id_, path in uploaded_tracks.items():
            if id_ not in track_ids:
                try:
                    tags = mutagen.File(path)  # type: ignore[attr-defined]
                    artists = tags["TPE1"]
                    title = tags["TIT2"]
                    write(f"Removing `{artists} - {title}`")
                except mutagen.mp3.HeaderNotFoundError:
                    write(f"Broken file `{path}`, removing")
                path.unlink()

    def arrange_files() -> None:
        for it in dest_folder.glob("*.mp3"):
            id_ = "_".join(it.stem.split("_")[1:])
            index = track_indices[id_]

            it.rename(it.with_stem(f"{index}_{id_}"))

    def download_missing_tracks() -> None:
        n_downloaded = len(list(dest_folder.glob("*.mp3")))

        for i, it in enumerate(missing_tracks):
            id_ = it.id
            index = track_indices[id_]

            write(f"[*{n_downloaded + i + 1}*/*{n_downloaded + len(missing_tracks)}*] Downloading `{it}`")
            try:
                with atomic_path(dest_folder / f"{index}_{id_}.mp3", suffix=".mp3") as tmp_path:
                    it.download(tmp_path)
                    with Status("Embedding metadata"):
                        it.saturated_metadata(database.search_track(str(it))).embed(tmp_path)
            except TrackNotAvailable as e:
                write(f"**Failed to download:** {e}")

    remove_extraneous_tracks()
    arrange_files()
    download_missing_tracks()


#
# Точка входа
#


def main() -> None:
    config = cache_file(pathlib.Path(".config"))
    database = Database()

    def vkontakte_routine() -> None:
        client = make_vkontakte_client(config, cache_file(pathlib.Path(".vkontakte")))
        user = client.user()
        tracks = user.tracks

        sync(tracks, MUSIC_FOLDER / "ВКонтакте", database)

    def yandex_music_routine() -> None:
        client = make_yandex_music_client(config)
        # Явно указываем пользователя на тот случай, когда отвалится токен,
        # и придётся зайти из-под чужого аккаунта
        user = client.user("lucruum666")
        tracks = user.tracks

        sync(tracks, MUSIC_FOLDER / "Яндекс Музыка", database)

    def youtube_routine() -> None:
        client = make_youtube_client(config)
        me = "АветисСехпосян"
        me = urllib.parse.quote(me)
        user = client.user(f"/@{me}")
        playlist = user.playlists[-1]
        videos = playlist.videos

        sync(videos, MUSIC_FOLDER / "YouTube", database)

    for routine in (vkontakte_routine, yandex_music_routine, youtube_routine):
        try:
            routine()
        except Exception:
            print(traceback.format_exc())
            pdb.post_mortem()


if __name__ == "__main__":
    import pdb

    ensure_ffmpeg_installed()
    patch_tqdm()
    patch_vk_api()
    main()
