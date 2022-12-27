import abc
import os
import pathlib
import re

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
# Файловая система
#


MUSIC_FOLDER = pathlib.Path(f"{os.environ['USERPROFILE']}") / "Music"


def remove_invalid_path_chars(s: str) -> str:
    """Удаляет из строки символы, недопустимые в именах путей"""
    return re.sub(r'[:?"*/\<>|]', "", s)


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
    def artists(self) -> str:
        return ", ".join(self._impl.artists_name())

    @property
    def title(self) -> str:
        return self._impl.title or ""

    def download(self, path: pathlib.Path) -> None:
        self._impl.download(str(path))


#
# Точка входа
#


def main() -> None:
    client = YandexMusicClient(os.environ["YANDEX_LOGIN"], os.environ["YANDEX_PASSWORD"])
    user = client.user()
    tracks = user.tracks

    for it in tracks:
        path = MUSIC_FOLDER / "Яндекс Музыка" / f"{remove_invalid_path_chars(str(it))}.mp3"

        if not path.exists():
            print(f"Downloading `{it}`...", flush=True, end="")
            it.download(path)
            print("\b\b\b, done")


if __name__ == "__main__":
    main()
