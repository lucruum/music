from typing import Any, Iterator, Optional, Sequence
import abc
import atexit
import contextlib
import functools
import html
import json
import os
import pathlib
import pickle
import re
import subprocess
import tempfile
import traceback
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
import ytmusicapi  # type: ignore[import]


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
    def __init__(self, cache: AutovivificiousDict) -> None:
        self._cache = cache
        self._genius_database = GeniusDatabase(cache)
        self._youtube_music_database = YouTubeMusicDatabase(cache)

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
        self._database = database
        self._query = query

    @property
    def album(self) -> str:
        if youtube_found := self._database._youtube_music_database.search_track(self._query):
            if youtube_found.album:
                return youtube_found.album
        if genius_found := self._database._genius_database.search_track(self._query):
            if genius_found.album:
                return genius_found.album
        return ""

    @property
    def artists(self) -> str:
        if youtube_found := self._database._youtube_music_database.search_track(self._query):
            if youtube_found.artists:
                return youtube_found.artists
        if genius_found := self._database._genius_database.search_track(self._query):
            if genius_found.artists:
                return genius_found.artists
        return ""

    @property
    def cover(self) -> bytes:
        if youtube_found := self._database._youtube_music_database.search_track(self._query):
            if youtube_found.cover:
                return youtube_found.cover
        if genius_found := self._database._genius_database.search_track(self._query):
            if genius_found.cover:
                return genius_found.cover
        return b""

    @property
    def lyrics(self) -> str:
        if youtube_found := self._database._youtube_music_database.search_track(self._query):
            if youtube_found.lyrics:
                return youtube_found.lyrics
        if genius_found := self._database._genius_database.search_track(self._query):
            if genius_found.lyrics:
                return genius_found.lyrics
        return ""

    @property
    def title(self) -> str:
        if youtube_found := self._database._youtube_music_database.search_track(self._query):
            if youtube_found.title:
                return youtube_found.title
        if genius_found := self._database._genius_database.search_track(self._query):
            if genius_found.title:
                return genius_found.title
        return ""


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

    def __init__(self, cache: AutovivificiousDict):
        self._cache = cache

    def search_track(self, query: str) -> Optional["GeniusDatabaseTrack"]:
        if query in self._cache["genius"]["tracks"]:
            if self._cache["genius"]["tracks"][query] is not None:
                return self._cache["genius"]["tracks"][query]  # type: ignore[no-any-return]
            return None

        response = requests.get("https://genius.com/api/search/song", params={"q": query})
        for it in response.json()["response"]["sections"][0]["hits"]:
            if it["result"]["artist_names"] not in GeniusDatabase.TRANSLATION_ARTIST_NAMES:
                track = GeniusDatabaseTrack(it["result"])
                self._cache["genius"]["tracks"][query] = track
                return track

        self._cache["genius"]["tracks"][query] = None
        return None


class GeniusDatabaseTrack:
    def __init__(self, impl: dict[str, Any]):
        self._album_url = f"https://genius.com/api/songs/{impl['id']}"
        self._cover_urls = (str(impl["header_image_url"]), str(impl["song_art_image_url"]))
        self._lyrics_url = str(impl["url"])
        self.artists = GeniusDatabaseTrack._strip_translation(impl["primary_artist"]["name"])
        self.title = GeniusDatabaseTrack._strip_translation(impl["title"])

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
                if element.name in ("inread-ad",):
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
        return re.sub("\\[.*?\\]\n+", "", s)


class YouTubeMusicDatabase:
    _IMPL = ytmusicapi.YTMusic()
    # Язык возвращаемых данных задаётся параметром `language` конструктора `YTMusic`,
    # но в виду того, что библиотека не поддерживает русскую локализацию,
    # выставляем язык хоста напрямую в заголовках
    # (см. https://github.com/sigma67/ytmusicapi/tree/master/ytmusicapi/locales#readme)
    _IMPL.context["context"]["client"]["hl"] = "ru"

    def __init__(self, cache: AutovivificiousDict):
        self._cache = cache

    def search_track(self, query: str) -> Optional["YouTubeMusicDatabaseTrack"]:
        if query in self._cache["youtube_music"]["tracks"]:
            if self._cache["youtube_music"]["tracks"][query] is not None:
                return self._cache["youtube_music"]["tracks"][query]  # type: ignore[no-any-return]
            return None

        if found := YouTubeMusicDatabase._IMPL.search(query, filter="songs", limit=1):
            track = YouTubeMusicDatabaseTrack(found[0])
            self._cache["youtube_music"]["tracks"][query] = track
            return track

        self._cache["youtube_music"]["tracks"][query] = None
        return None


class YouTubeMusicDatabaseTrack:
    def __init__(self, impl: dict[str, Any]):
        self._cover_url = str(impl["thumbnails"][0]["url"]).replace("w60-h60", "w600-h600")
        self._video_id = str(impl["videoId"])
        self.album = str(impl["album"]["name"]) if impl["album"] is not None else ""
        self.artists = ", ".join(it["name"] for it in impl["artists"])
        self.title = str(impl["title"])

    @property
    def cover(self) -> bytes:
        return requests.get(self._cover_url).content

    @property
    def lyrics(self) -> str:
        try:
            playlist = YouTubeMusicDatabase._IMPL.get_watch_playlist(self._video_id, limit=1)
            result = YouTubeMusicDatabase._IMPL.get_lyrics(playlist["lyrics"])["lyrics"]
            return result is not None and str(result) or ""
        except Exception:
            return ""


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
        try:
            config["vkontakte"]["credentials"] = (login, password)
            return VKontakteClient(login, password, cache)
        except vk_api.exceptions.BadPassword:
            write("Invalid login or password")
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

            for wrappee in tqdm.tqdm(
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
        )

    def download(self, path: pathlib.Path) -> None:
        class Bar(tqdm.tqdm):  # type: ignore[type-arg]
            def __init__(self, **kwargs: dict[str, Any]):
                assert isinstance(kwargs["total"], int)

                super().__init__(ascii=".:", desc="Receiving track", total=kwargs["total"])

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

        self._client = client
        self.id = str(info["uid"])
        self.name = str(info["name"])

    def show(self) -> str:
        return self.name

    @property
    def tracks(self) -> list["YandexMusicTrack"]:
        return [YandexMusicTrack(it) for it in self._client._impl.users_likes_tracks(self.id).fetch_tracks()]


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

    def saturated_metadata(self, suggestion: DatabaseTrack) -> "TrackMetadata":
        return TrackMetadata(
            album=self.album,
            artists=self.artists,
            cover=self.cover,
            lyrics=self.lyrics or suggestion.lyrics,
            title=self.title,
        )

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
            with path.open("wb") as file:
                for data in response.iter_content(chunk_size=1024):
                    bar.update(file.write(data))


#
# Общее
#


class TrackMetadata:
    def __init__(self, *, album: str, artists: str, cover: bytes, lyrics: str, title: str):
        self.album = album
        self.artists = artists
        self.cover = cover
        self.lyrics = lyrics
        self.title = title

    def embed(self, path: pathlib.Path) -> None:
        tags = mutagen.id3.ID3()  # type: ignore[no-untyped-call]
        tags["TALB"] = mutagen.id3.TALB(text=self.album)  # type: ignore[attr-defined]
        tags["TPE1"] = mutagen.id3.TPE1(text=self.artists)  # type: ignore[attr-defined]
        tags["APIC"] = mutagen.id3.APIC(data=self.cover)  # type: ignore[attr-defined]
        tags["USLT"] = mutagen.id3.USLT(text=self.lyrics)  # type: ignore[attr-defined]
        tags["TIT2"] = mutagen.id3.TIT2(text=self.title)  # type: ignore[attr-defined]
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
                write("Embedding metadata...", flush=True, end="")
                it.saturated_metadata(database.search_track(str(it))).embed(tmp_path)
                write("\b\b\b, done")

    remove_extraneous_tracks()
    arrange_files()
    download_missing_tracks()


#
# Точка входа
#


def main() -> None:
    config = cache_file(pathlib.Path(".config"))
    database = Database(cache_file(pathlib.Path(".database")))

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
