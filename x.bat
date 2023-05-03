@echo off

@REM Одиночный cls в Windows Terminal'е не всегда полностью очищает экран,
@REM из-за чего вывод предыдущей команды наслаивается на вывод текущей
set "command=          cls & cls &"
set "command=%command% black music.py --line-length=120 &"
set "command=%command% flake8 music.py --max-line-length=120 &"
set "command=%command% mypy music.py --pretty --show-error-context --strict &"
set "command=%command% python -m doctest music.py &"
set "command=%command% python music.py"

python -m on_touch music.py "%command%"
