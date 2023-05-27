@echo off

@REM Одиночный cls в Windows Terminal'е не всегда полностью очищает экран,
@REM из-за чего вывод предыдущей команды наслаивается на вывод текущей
set "command=          cls & cls &"
set "command=%command% poetry run black music.py &"
set "command=%command% poetry run flake8 music.py &"
set "command=%command% poetry run mypy music.py &"
set "command=%command% poetry run python -m doctest music.py &"
set "command=%command% poetry run python music.py"

python -m on_touch music.py "%command%"
