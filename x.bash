#!/usr/bin/env bash

YANDEX_MUSIC_PY_TYPED_PATH="$(poetry env info --path)/lib/python3.11/site-packages/yandex_music/py.typed"

# Сообщаем mypy, что библиотека содержит описания типов
if [[ ! -f $YANDEX_MUSIC_PY_TYPED_PATH ]]; then
    touch $YANDEX_MUSIC_PY_TYPED_PATH
fi

poetry run python tools/watch.py music.py "
    clear;
    poetry run black music.py;
    poetry run flake8 music.py;
    poetry run mypy music.py;
    poetry run python -m doctest music.py;
    poetry run python music.py;
"
