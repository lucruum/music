import functools
import itertools
import os
import yandex_music

yandex_music.Client.notice_displayed = True

class Wrapper:
    def __init__(self, handle):
        self.handle = handle

    def __eq__(self, other):
        return self.handle == other.handle

class Track(Wrapper):
    @property
    def id(self):
        return self.handle.id

    @property
    def name(self):
        return self.handle.title

    @functools.cached_property
    def albums(self):
        result = self.handle.albums
        result = map(Album, result)
        return list(result)

    @functools.cached_property
    def artists(self):
        result = self.handle.artists
        result = map(Artist, result)
        return list(result)

class Album(Wrapper):
    @property
    def id(self):
        return self.handle.id

    @property
    def name(self):
        return self.handle.title

    @functools.cached_property
    def tracks(self):
        result = self.handle.with_tracks().volumes
        result = itertools.chain.from_iterable(result)
        result = map(Track, result)
        return list(result)

    @functools.cached_property
    def artists(self):
        result = self.handle.artists
        result = map(Artist, result)
        return list(result)

class Artist(Wrapper):
    @property
    def id(self):
        return self.handle.id

    @property
    def name(self):
        return self.handle.name

    @functools.cached_property
    def tracks(self):
        n_tracks = self.handle.counts.tracks
        result = self.handle.get_tracks(page_size=n_tracks)
        result = map(Track, result)
        return list(result)

    @functools.cached_property
    def albums(self):
        n_albums = self.handle.counts.direct_albums
        result = self.handle.get_albums(page_size=n_albums)
        result = map(Album, result)
        return list(result)

class Playlist(Wrapper):
    @property
    def id(self):
        return self.handle.id

    @property
    def name(self):
        return self.handle.title

    @functools.cached_property
    def tracks(self):
        result = self.handle.fetch_tracks()
        result = map(lambda x: x.track, result)
        result = map(Track, result)
        return list(result)

class Client(Wrapper):
    @classmethod
    def from_token(cls, token):
        result = yandex_music.Client(token, report_new_fields=False)
        return Client(result)

    @property
    def id(self):
        return self.handle.me.account.uid

    @functools.cached_property
    def tracks(self):
        result = self.handle.users_likes_tracks()
        result = result.fetch_tracks()
        result = map(Track, result)
        return list(result)

    @functools.cached_property
    def albums(self):
        result = self.handle.users_likes_albums()
        result = map(lambda x: x.album, result)
        result = map(Album, result)
        return list(result)

    @functools.cached_property
    def artists(self):
        result = self.handle.users_likes_artists()
        result = map(lambda x: x.artist, result)
        result = map(Artist, result)
        return list(result)

    @functools.cached_property
    def playlists(self):
        result = self.handle.users_playlists_list()
        favorite = self.handle.users_playlists(3)
        favorite.title = 'Favorites'
        result = [favorite] + result
        result = map(Playlist, result)
        return list(result)

client = Client.from_token(os.environ['MUSIC_TOKEN'])