import colorama
import functools
import itertools
import mutagen.id3
import mutagen.mp3
import os
import pathlib
import requests
import sys
import tqdm
import yandex_music

colorama.init(autoreset=True)
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

    @functools.cached_property
    def cover(self):
        uri = f'https://{self.handle.cover_uri.replace("%%", "400x400")}'
        result = requests.get(uri)
        return result.content

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

def search(client, query):
    result = client.handle.search(query)
    result = result.best
    klass = {
        'track': Track,
        'album': Album,
        'artist': Artist,
        'playlist': Playlist,
    }
    return klass[result.type](result.result)

def sync(client, folder, tracks):
    def message(color, status, track):
        width, _ = os.get_terminal_size()
        name = track.name
        artists = ', '.join(it.name for it in track.artists)
        output = f'{{}}{{}}[{status}]{{}} {name} - {artists}'
        output_without_escapes = output.format('', '', '')
        print(output.format('\r', color, colorama.Fore.RESET) + ' ' * (width - len(output_without_escapes)))

    def compute_local_tracks():
        return list(folder.glob('*.mp3'))

    def compute_mounts(local_tracks):
        remote_tracks = [''] + [it.stem.split('_')[-1] for it in local_tracks]
        remote_tracks = client.handle.tracks(remote_tracks)
        remote_tracks = map(Track, remote_tracks)
        return dict(zip(local_tracks, remote_tracks))

    def compute_redundant_tracks(local_tracks, mounts):
        return [it for it in local_tracks if mounts[it] not in tracks]

    def compute_non_fetched_tracks(local_tracks, mounts):
        return [it for it in tracks if it not in mounts.values()]

    def compute_order(local_tracks, mounts):
        return {path: tracks.index(mount) for path, mount in mounts.items()}

    def download_impl(track, path):
        url = track.handle.get_download_info(get_direct_links=True)
        url = url[0].direct_link

        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        with tqdm.tqdm(
            desc=track.name,
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]'
        ) as bar, \
        open(path, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

    def download(track):
        path = folder / f'{track.id}.mp3'
        download_impl(track, path)

        tags = mutagen.mp3.EasyMP3(path)
        tags['title'] = track.name
        tags['artist'] = ', '.join(it.name for it in track.artists)
        tags['album'] = ', '.join(it.name for it in track.albums)
        tags.save()

        tags = mutagen.id3.ID3(path)
        tags['APIC'] = mutagen.id3.APIC(
            encoding=3,
            mime='image/jpeg',
            type=3,
            desc=u'Cover',
            data=track.cover
        )
        tags.save()

    def main():
        folder.mkdir(exist_ok=True)

        local_tracks = compute_local_tracks()
        mounts = compute_mounts(local_tracks)

        for path in compute_redundant_tracks(local_tracks, mounts):
            path.unlink()
            message(colorama.Fore.CYAN, '-', mounts[path])

        for track in tqdm.tqdm(compute_non_fetched_tracks(local_tracks, mounts), leave=False):
            try:
                download(track)
                message(colorama.Fore.GREEN, '+', track)
            except yandex_music.exceptions.Unauthorized:
                message(colorama.Fore.RED, '>', track)

        local_tracks = compute_local_tracks()
        mounts = compute_mounts(local_tracks)
        for path, index in compute_order(local_tracks, mounts).items():
            path.rename(folder / f'{index}_{path.stem.split("_")[-1]}.mp3')

    return main()

client = Client.from_token(os.environ['MUSIC_TOKEN'])
folder = {
    'linux':  pathlib.Path('/storage/emulated/0/Music'),
    'win32': pathlib.Path('C:/Users/naoh4/Music'),
}[sys.platform]