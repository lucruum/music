from typing import Any, Iterator, Protocol, Sequence
import abc
import atexit
import contextlib
import os
import pathlib
import pickle
import re
import subprocess
import tempfile
import uuid
import warnings

import bs4
import colorama
import ffpb  # type: ignore[import]
import mutagen.id3
import requests
import tqdm
import vk_api  # type: ignore[import]
import vk_api.audio  # type: ignore[import]
import yandex_music


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


def write(*objs: Any, end: str = "\n", flush: bool = True, sep: str = " ") -> None:
    """Стилизованный вывод"""
    output = sep.join(map(str, objs))
    output = markup(output)
    output = fit(output, os.get_terminal_size()[0], f"{colorama.Style.RESET_ALL}…")
    print(output, end=end, flush=flush)


#
# Файловая система
#


MUSIC_FOLDER = pathlib.Path(f"{os.environ['USERPROFILE']}") / "Music"


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
        tmp_path.rename(path)
    finally:
        tmp_path.unlink(missing_ok=True)


#
# Кэширование
#


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
# ВКонтакте
#


class VKontakteClient:
    def __init__(self, login: str, password: str):
        session = vk_api.VkApi(
            login,
            password,
            captcha_handler=lambda x: x.try_again(input(f"Enter symbols from the picture {x.get_url()}: ")),
        )
        session.auth()

        self._api = session.get_api()
        self._audio = vk_api.audio.VkAudio(session)

    def user(self, id_: str | int | None = None) -> "VKontakteUser":
        return VKontakteUser(self, id_)


def make_vkontakte_client(config: AutovivificiousDict) -> VKontakteClient:
    """Создание клиента с ранее введёнными учётными данными"""
    if "credentials" in config["vkontakte"]:
        login, password = config["vkontakte"]["credentials"]
    else:
        login = input("VKontakte login: ")
        password = input(f"{login.split('@')[0]}'s password: ")

    while True:
        try:
            config["vkontakte"]["credentials"] = (login, password)
            return VKontakteClient(login, password)
        except vk_api.exceptions.BadPassword:
            write("Invalid login or password")
            login = input("VKontakte login: ")
            password = input(f"{login.split('@')[0]}'s password: ")


class VKontakteUser(Show):
    def __init__(self, client: VKontakteClient, id_: str | int | None):
        info = client._api.users.get(user_ids=id_)[0]

        self.client = client
        self.full_name = f"{info['first_name']} {info['last_name']}"
        self.id = str(info["id"])

    def show(self) -> str:
        return self.full_name

    @property
    def tracks(self) -> list["VKontakteTrack"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=bs4.MarkupResemblesLocatorWarning)
            return [VKontakteTrack(it) for it in self.client._audio.get_iter(self.id)]


class VKontakteTrack(Show):
    def __init__(self, impl: dict[str, Any]):
        self.artists = impl["artist"]
        # Обложки отсортированы по возрастанию расширения
        self.cover_url = impl["track_covers"] and str(impl["track_covers"][-1]) or None
        self.id = f"{impl['owner_id']}{impl['id']}"
        self.title = impl["title"]
        self.url = impl["url"]

    def show(self) -> str:
        return f"{self.artists} - {self.title}"

    @property
    def cover(self) -> bytes:
        if self.cover_url:
            return requests.get(self.cover_url).content
        return b""

    def download(self, path: pathlib.Path) -> None:
        class Bar(tqdm.tqdm):  # type: ignore[type-arg]
            def __init__(self, **kwargs: dict[str, Any]):
                assert isinstance(kwargs["total"], int)

                super().__init__(ascii=".:", desc="Receiving track", total=kwargs["total"])

        with ffpb.ProgressNotifier(tqdm=Bar) as notifier:
            with atomic_path(path, suffix=".mp3") as tmp_path:
                process = subprocess.Popen(
                    ["ffmpeg", "-http_persistent", "false", "-i", self.url, "-codec", "copy", tmp_path],
                    stderr=subprocess.PIPE,
                )

                while True:
                    if stream := process.stderr:
                        if data := stream.read(1):
                            notifier(data)
                        elif process.poll() is not None:
                            break

                tags = mutagen.id3.ID3()  # type: ignore[no-untyped-call]
                tags["TPE1"] = mutagen.id3.TPE1(text=self.artists)  # type: ignore[attr-defined]
                tags["APIC"] = mutagen.id3.APIC(data=self.cover)  # type: ignore[attr-defined]
                tags["TIT2"] = mutagen.id3.TIT2(text=self.title)  # type: ignore[attr-defined]
                tags.save(tmp_path)


#
# Яндекс Музыка
#


# Не показывать сообщение с лицензией
yandex_music.Client.notice_displayed = True


class YandexMusicClient:
    def __init__(self, login: str, password: str):
        self._impl = yandex_music.Client.from_credentials(login, password, report_new_fields=False)

    def user(self, id_: str | None = None) -> "YandexMusicUser":
        return YandexMusicUser(self, id_)


def make_yandex_music_client(config: AutovivificiousDict) -> YandexMusicClient:
    """Создание клиента с ранее введёнными учётными данными"""
    if "credentials" in config["yandex_music"]:
        login, password = config["yandex_music"]["credentials"]
    else:
        login = input("Yandex Music login: ")
        password = input(f"{login.split('@')[0]}'s password: ")

    while True:
        try:
            config["yandex_music"]["credentials"] = (login, password)
            return YandexMusicClient(login, password)
        except yandex_music.exceptions.BadRequest:
            write("Invalid login or password")
            login = input("Yandex Music login: ")
            password = input(f"{login.split('@')[0]}'s password: ")


class YandexMusicUser(Show):
    def __init__(self, client: YandexMusicClient, id_: str | None):
        info_url = f"{client._impl.base_url}/users/{id_}"
        info = client._impl._request.get(info_url)

        self.client = client
        self.id = str(info["uid"])
        self.name = str(info["name"])

    def show(self) -> str:
        return self.name

    @property
    def tracks(self) -> list["YandexMusicTrack"]:
        return [YandexMusicTrack(it) for it in self.client._impl.users_likes_tracks(self.id).fetch_tracks()]


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

    def download(self, path: pathlib.Path) -> None:
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
            with atomic_path(path) as tmp_path:
                with tmp_path.open("wb") as file:
                    for data in response.iter_content(chunk_size=1024):
                        bar.update(file.write(data))

                tags = mutagen.id3.ID3()  # type: ignore[no-untyped-call]
                tags["TALB"] = mutagen.id3.TALB(text=self.album)  # type: ignore[attr-defined]
                tags["TPE1"] = mutagen.id3.TPE1(text=self.artists)  # type: ignore[attr-defined]
                tags["APIC"] = mutagen.id3.APIC(data=self.cover)  # type: ignore[attr-defined]
                tags["USLT"] = mutagen.id3.USLT(text=self.lyrics)  # type: ignore[attr-defined]
                tags["TIT2"] = mutagen.id3.TIT2(text=self.title)  # type: ignore[attr-defined]
                tags.save(tmp_path)


#
# Общее
#


class Downloadable(Protocol):
    @property
    def id(self) -> str:
        pass

    def download(self, path: pathlib.Path) -> None:
        pass


def sync(src_tracks: Sequence[Downloadable], dest_folder: pathlib.Path) -> None:
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
            it.download(dest_folder / f"{index}_{id_}.mp3")

    remove_extraneous_tracks()
    arrange_files()
    download_missing_tracks()


#
# Точка входа
#


def main() -> None:
    config = cache_file(pathlib.Path(".config"))

    def vkontakte_routine() -> None:
        client = make_vkontakte_client(config)
        user = client.user()
        tracks = user.tracks

        sync(tracks, MUSIC_FOLDER / "ВКонтакте")

    def yandex_music_routine() -> None:
        client = make_yandex_music_client(config)
        user = client.user()
        tracks = user.tracks

        sync(tracks, MUSIC_FOLDER / "Яндекс Музыка")

    vkontakte_routine()
    yandex_music_routine()


if __name__ == "__main__":
    main()
