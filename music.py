from types import NoneType, TracebackType
from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    NoReturn,
    Optional,
    overload,
    Sequence,
    TYPE_CHECKING,
    Type,
    TypeVar,
)

if TYPE_CHECKING:
    from _typeshed import SupportsWrite

import abc
import atexit
import contextlib
import difflib
import functools
import hashlib
import html
import json
import os
import pathlib
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import warnings

import bs4
import colorama
import ffpb  # type: ignore[import]
import fp.fp  # type: ignore[import]
import mutagen.id3
import requests
import tqdm
import vk_api  # type: ignore[import]
import vk_api.audio  # type: ignore[import]
import yandex_music
import ytmusicapi  # type: ignore[import]


#
# Типы
#


if TYPE_CHECKING:
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


if TYPE_CHECKING:

    @overload
    def mytqdm(
        iterable: Iterable[T],
        desc: str | None = ...,
        total: float | None = ...,
        leave: bool | None = ...,
        file: SupportsWrite[str] | None = ...,
        ncols: int | None = ...,
        mininterval: float = ...,
        maxinterval: float = ...,
        miniters: float | None = ...,
        ascii: bool | str | None = ...,
        disable: bool | None = ...,
        unit: str = ...,
        unit_scale: bool | float = ...,
        dynamic_ncols: bool = ...,
        smoothing: float = ...,
        bar_format: str | None = ...,
        initial: float = ...,
        position: int | None = ...,
        postfix: Mapping[str, object] | str | None = ...,
        unit_divisor: float = ...,
        write_bytes: bool | None = ...,
        lock_args: tuple[bool | None, float | None] | tuple[bool | None] | None = ...,
        nrows: int | None = ...,
        colour: str | None = ...,
        delay: float | None = ...,
        gui: bool = ...,
        **kwargs: dict[str, Any],
    ) -> tqdm.tqdm[T]:
        ...

    @overload
    def mytqdm(
        iterable: None = ...,
        desc: str | None = ...,
        total: float | None = ...,
        leave: bool | None = ...,
        file: SupportsWrite[str] | None = ...,
        ncols: int | None = ...,
        mininterval: float = ...,
        maxinterval: float = ...,
        miniters: float | None = ...,
        ascii: bool | str | None = ...,
        disable: bool | None = ...,
        unit: str = ...,
        unit_scale: bool | float = ...,
        dynamic_ncols: bool = ...,
        smoothing: float = ...,
        bar_format: str | None = ...,
        initial: float = ...,
        position: int | None = ...,
        postfix: Mapping[str, object] | str | None = ...,
        unit_divisor: float = ...,
        write_bytes: bool | None = ...,
        lock_args: tuple[bool | None, float | None] | tuple[bool | None] | None = ...,
        nrows: int | None = ...,
        colour: str | None = ...,
        delay: float | None = ...,
        gui: bool = ...,
        **kwargs: dict[str, Any],
    ) -> tqdm.tqdm[NoReturn]:
        ...

    def mytqdm(
        iterable: Iterable[T] | None = ...,
        desc: str | None = ...,
        total: float | None = ...,
        leave: bool | None = ...,
        file: SupportsWrite[str] | None = ...,
        ncols: int | None = ...,
        mininterval: float = ...,
        maxinterval: float = ...,
        miniters: float | None = ...,
        ascii: bool | str | None = ...,
        disable: bool | None = ...,
        unit: str = ...,
        unit_scale: bool | float = ...,
        dynamic_ncols: bool = ...,
        smoothing: float = ...,
        bar_format: str | None = ...,
        initial: float = ...,
        position: int | None = ...,
        postfix: Mapping[str, object] | str | None = ...,
        unit_divisor: float = ...,
        write_bytes: bool | None = ...,
        lock_args: tuple[bool | None, float | None] | tuple[bool | None] | None = ...,
        nrows: int | None = ...,
        colour: str | None = ...,
        delay: float | None = ...,
        gui: bool = ...,
        **kwargs: dict[str, Any],
    ) -> tqdm.tqdm[T | NoReturn]:
        ...

else:

    def mytqdm(*args, **kwargs):
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
        if len(args) >= 6 and args[5] is None:
            args[5] = os.get_terminal_size()[0]
        if "ncols" not in kwargs or kwargs["ncols"] is None:
            kwargs["ncols"] = os.get_terminal_size()[0]
        return tqdm.tqdm(*args, **kwargs)


#
# Файловая система
#


MUSIC_FOLDER = (
    pathlib.Path(os.environ["USERPROFILE"], "Music")
    if sys.platform == "win32"
    else pathlib.Path("/storage/emulated/0/Music")
)


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


__proxy = ""


def proxy() -> str:
    """Адрес прокси-сервера"""
    if not __proxy:
        next_proxy()

    return __proxy


def next_proxy() -> None:
    """Получение нового адреса прокси-сервера"""
    global __proxy

    while True:
        # Элитные прокси-сервера реже отваливаются
        # Перемешиваем список прокси-серверов, чтобы не попасть в бесконечный цикл
        with Status("Searching for proxy server") as status:
            try:
                __proxy = fp.fp.FreeProxy(elite=True, rand=True).get()
            except fp.errors.FreeProxyException:
                status.fail("not found")
                continue

        # Проверка ответа сервера
        with Status(f"Trying to connect to `{__proxy}`") as status:
            try:
                response = requests.get(
                    "https://f4.bcbits.com/img/a1056493284_10.jpg",
                    proxies={"https": __proxy},
                    timeout=5,
                )
                actual_hashsum = hashlib.sha256(response.content).hexdigest()
                expected_hashsum = "f9e2c765115cfc602faace1485f86c3507d8e246471cf126dcd0b647df04368f"
                if actual_hashsum != expected_hashsum:
                    status.fail("invalid server response")
                    continue
            except requests.exceptions.RequestException as e:
                status.fail(str(e))
                continue

        # Проверка скорости соединения
        with Status("Testing connection speed") as status:
            try:
                start = time.perf_counter()
                response = requests.get(
                    "http://ipv4.download.thinkbroadband.com:80/2MB.zip",
                    proxies={"http": __proxy},
                    stream=True,
                )
                length = int(response.headers["content-length"])
                got = 0
                for data in response.iter_content(1024):
                    got += len(data)
                    done = int((got / length) * 20)
                    bar = f"[*{'=' * done}{' ' * (20 - done)}*]"
                    speed = f"`{got / (time.perf_counter() - start) / 1024 ** 2:.2f}` MBps"
                    status.set_message(f"Testing connection speed, {bar} {speed}")
                if time.perf_counter() - start > 10:
                    # Соединение медленнее 0.1 МБ/c
                    status.fail("too slow")
                    continue
            except requests.exceptions.RequestException as e:
                status.fail(str(e))
                continue

        return


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
        query = re.sub(r"\s*[\{\[\(].*?[\)\]\}]", "", query)
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

        artists_artists_ratio = difflib.SequenceMatcher(a=query.split(" - ")[0], b=found.split(" - ")[0]).ratio()
        title_title_ratio = difflib.SequenceMatcher(a=query.split(" - ")[1], b=found.split(" - ")[1]).ratio()

        artists_title_ratio = difflib.SequenceMatcher(a=query.split(" - ")[0], b=found.split(" - ")[1]).ratio()
        title_artists_ratio = difflib.SequenceMatcher(a=query.split(" - ")[1], b=found.split(" - ")[0]).ratio()

        return max(artists_artists_ratio + title_title_ratio, artists_title_ratio + title_artists_ratio) / 2

    @property
    def album(self) -> str:
        for it in self._matches:
            if it is not None and it.album:
                return it.album
        return ""

    @property
    def artists(self) -> str:
        for it in self._matches:
            if it is not None and it.artists:
                return it.artists
        return ""

    @property
    def cover(self) -> bytes:
        for it in self._matches:
            if it is not None and it.cover:
                return it.cover
        return b""

    @property
    def lyrics(self) -> str:
        for it in self._matches:
            if it is not None and it.lyrics:
                return it.lyrics
        return ""

    @property
    def title(self) -> str:
        for it in self._matches:
            if it is not None and it.title:
                return it.title
        return ""

    @property
    def year(self) -> str:
        for it in self._matches:
            if it is not None and it.year:
                return it.year
        return ""


class BandcampDatabase:
    def search_track(self, query: str) -> Optional["BandcampDatabaseTrack"]:
        response = requests.get("https://bandcamp.com/search", params={"q": query, "item_type": "t"})
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
        response = requests.get(self._url)
        html_ = response.text
        soup = bs4.BeautifulSoup(html_, "html.parser")

        popup_image = soup.find("a", class_="popupImage")
        assert isinstance(popup_image, bs4.Tag)
        cover_url = popup_image["href"]
        assert isinstance(cover_url, str)

        while True:
            try:
                return requests.get(cover_url, proxies={"https": proxy()}).content
            except requests.exceptions.RequestException:
                next_proxy()


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
        result = GeniusDatabaseTrack._remove_section_headers(result)
        return result

    @staticmethod
    def _scrape_text(element: bs4.element.PageElement) -> str:
        """Извлечение текста с учётом элементов переноса строк"""

        def impl(element: bs4.element.PageElement) -> list[str]:
            if isinstance(element, bs4.element.NavigableString):
                return [str(element)]
            elif isinstance(element, bs4.element.Tag):
                if element.name in ("inread-ad", "primis-player"):
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

    @staticmethod
    def _remove_section_headers(s: str) -> str:
        """Удаление '[Текст песни «…»]', '[Куплет]' и т.п. из текста песни"""
        if re.fullmatch(r"\[.*?\]", s):
            return ""
        return re.sub("\\[.*?\\]\n+", "", s)


class YouTubeMusicDatabase:
    def __init__(self) -> None:
        self._impl = ytmusicapi.YTMusic()
        # Язык возвращаемых данных задаётся параметром `language` конструктора `YTMusic`,
        # но в виду того, что библиотека не поддерживает русскую локализацию,
        # выставляем язык хоста напрямую в заголовках
        # (см. https://github.com/sigma67/ytmusicapi/tree/master/ytmusicapi/locales#readme)
        self._impl.context["context"]["client"]["hl"] = "ru"

    def search_track(self, query: str) -> Optional["YouTubeMusicDatabaseTrack"]:
        if found := self._impl.search(query, filter="songs", limit=1):
            return YouTubeMusicDatabaseTrack(self, found[0])
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
            result = self._database._impl.get_lyrics(self._playlist["lyrics"])["lyrics"]
            return result is not None and str(result) or ""
        except Exception:
            return ""

    @property
    def year(self) -> str:
        return str(self._playlist["tracks"][0].get("year", ""))


#
# ВКонтакте
#


class VKontakteClient:
    def __init__(self, login: str, password: str, cache: AutovivificiousDict):
        session = vk_api.VkApi(
            login,
            password,
            captcha_handler=lambda x: x.try_again(input(f"Enter symbols from the picture {x.get_url()}: ")),
        )
        session.auth()

        info = session.get_api().users.get(user_ids=None)[0]

        self._api = session.get_api()
        self._audio = vk_api.audio.VkAudio(session)
        self._cache = cache
        self.id = str(info["id"])

    def group(self, id_: str | int) -> "VKontakteGroup":
        return VKontakteGroup(self, id_)

    def user(self, id_: str | int | None = None) -> "VKontakteUser":
        return VKontakteUser(self, id_)


def make_vkontakte_client(config: AutovivificiousDict, cache: AutovivificiousDict) -> VKontakteClient:
    """Создание клиента с ранее введёнными учётными данными"""
    if "credentials" in config["vkontakte"]:
        login, password = config["vkontakte"]["credentials"]
    else:
        login = input("VKontakte login: ")
        password = input(f"{login.split('@')[0]}'s password: ")

    while True:
        with Status("Logging in to VKontakte") as status:
            config["vkontakte"]["credentials"] = (login, password)
            try:
                return VKontakteClient(login, password, cache)
            except (vk_api.exceptions.BadPassword, vk_api.exceptions.LoginRequired, vk_api.exceptions.PasswordRequired):
                status.fail("invalid login or password")
                login = input("VKontakte login: ")
                password = input(f"{login.split('@')[0]}'s password: ")


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
        return list(_VKontakteUserTracks(self))

    @property
    def wall(self) -> "VKontakteWall":
        return VKontakteWall(self._client, self.id)


class _VKontakteUserTracks:
    def __init__(self, user: VKontakteUser):
        self._user = user

    def __iter__(self) -> Iterator["VKontakteTrack"]:
        hashes = self._hashes()

        missing_hashes = []
        for it in hashes:
            id_ = "".join(it[:2])
            if id_ not in self._user._client._cache["user_tracks"][self._user.id]:
                missing_hashes.append(it)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=bs4.MarkupResemblesLocatorWarning)

            for wrappee in mytqdm(
                vk_api.audio.scrap_tracks(
                    missing_hashes,
                    int(self._user._client.id),
                    self._user._client._audio._vk.http,
                    convert_m3u8_links=self._user._client._audio.convert_m3u8_links,
                ),
                ascii=".:",
                desc="Fetching tracks",
                disable=len(missing_hashes) == 0,
                total=len(missing_hashes),
            ):
                track = VKontakteTrack(wrappee)
                self._user._client._cache["user_tracks"][self._user.id][track.id] = track

        for it in hashes:
            id_ = "".join(it[:2])
            yield self._user._client._cache["user_tracks"][self._user.id][id_]

    def _hashes(self) -> list[tuple[str, str, str, str]]:
        result = []

        # См. vk_api/audio.py:VkAudio.get_iter
        offset = 0
        while True:
            response = self._user._client._audio._vk.http.post(
                "https://m.vk.com/audio",
                data={
                    "act": "load_section",
                    "owner_id": self._user.id,
                    "playlist_id": -1,
                    "offset": offset,
                    "type": "playlist",
                    "access_hash": None,
                    "is_loading_all": 1,
                },
                allow_redirects=False,
            ).json()

            if not response["data"][0]:
                return []

            result.extend(vk_api.audio.scrap_ids(response["data"][0]["list"]))

            if response["data"][0]["hasMore"]:
                offset += vk_api.audio.TRACKS_PER_USER_PAGE
            else:
                break

        return result


class VKontakteWall:
    def __init__(self, client: VKontakteClient, owner_id: str):
        self._client = client
        self.owner_id = owner_id

    @property
    def posts(self) -> list["VKontaktePost"]:
        result: list[VKontaktePost] = []
        n_posts = self._client._api.wall.get(owner_id=self.owner_id)["count"]

        with mytqdm(
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
        # Обложки отсортированы по возрастанию расширения
        self._cover_url = impl["track_covers"] and str(impl["track_covers"][-1]) or None
        self._url = impl["url"]
        self.artists = impl["artist"]
        self.id = f"{impl['owner_id']}{impl['id']}"
        self.title = impl["title"]

    def show(self) -> str:
        return f"{self.artists} - {self.title}"

    @property
    def cover(self) -> bytes:
        if self._cover_url:
            return requests.get(self._cover_url).content
        return b""

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
        class Bar(tqdm.tqdm):  # type: ignore[type-arg]
            def __init__(self, **kwargs: dict[str, Any]):
                assert isinstance(kwargs["total"], int)

                super().__init__(
                    ascii=".:",
                    desc="Receiving track",
                    # См. комментарий к `mytqdm`
                    ncols=os.get_terminal_size()[0],
                    total=kwargs["total"],
                )

        with ffpb.ProgressNotifier(tqdm=Bar) as notifier:
            process = subprocess.Popen(
                ["ffmpeg", "-http_persistent", "false", "-i", self._url, "-codec", "copy", path],
                stderr=subprocess.PIPE,
            )

            while True:
                if stream := process.stderr:
                    if data := stream.read(1):
                        notifier(data)
                    elif process.poll() is not None:
                        break


class VKontakteAttachedTrack(VKontakteTrack):
    def __init__(self, impl: dict[str, Any]):
        self._cover_url = impl["coverUrl"] is not None and str(impl["coverUrl"]) or ""
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
        token = input("Yandex Music token: ")

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
                token = input("Yandex Music token: ")


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
        url = self._impl.get_download_info()[0].get_direct_link()
        response = requests.get(url, stream=True)
        length = int(response.headers["content-length"])

        with mytqdm(
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


#
# Общее
#


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
    src_tracks: Sequence[VKontakteTrack | YandexMusicTrack],
    dest_folder: pathlib.Path,
    database: Database,
) -> None:
    """Односторонняя синхронизация папки с треками"""
    track_ids = {it.id for it in src_tracks}
    track_indices = {it.id: i for i, it in enumerate(src_tracks)}
    uploaded_tracks = {it.stem.split("_")[-1]: it for it in dest_folder.glob("*.mp3")}
    missing_tracks = [it for it in src_tracks if it.id not in uploaded_tracks]

    def remove_extraneous_tracks() -> None:
        for id_, path in uploaded_tracks.items():
            if id_ not in track_ids:
                tags = mutagen.File(path)  # type: ignore[attr-defined]
                artists = tags["TPE1"]
                title = tags["TIT2"]

                write(f"Removing `{artists} - {title}`")
                path.unlink()

    def arrange_files() -> None:
        for it in dest_folder.glob("*.mp3"):
            id_ = it.stem.split("_")[-1]
            index = track_indices[id_]

            it.rename(it.with_stem(f"{index}_{id_}"))

    def download_missing_tracks() -> None:
        for i, it in enumerate(missing_tracks):
            id_ = it.id
            index = track_indices[id_]

            write(f"[*{i + 1}*/*{len(missing_tracks)}*] Downloading `{it}`")
            with atomic_path(dest_folder / f"{index}_{id_}.mp3", suffix=".mp3") as tmp_path:
                it.download(tmp_path)
                with Status("Embedding metadata"):
                    it.saturated_metadata(database.search_track(str(it))).embed(tmp_path)

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
        user = client.user()
        tracks = user.tracks

        sync(tracks, MUSIC_FOLDER / "Яндекс Музыка", database)

    vkontakte_routine()
    yandex_music_routine()


if __name__ == "__main__":
    import pdb

    try:
        main()
    except Exception:
        print(traceback.format_exc())
        pdb.post_mortem()
