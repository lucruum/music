set command= ^
    cls ^& ^
    black music.py --line-length=120 ^& ^
    flake8 music.py --max-line-length=120 ^& ^
    mypy music.py --strict ^& ^
    python -m doctest music.py ^& ^
    python music.py

python -m on_touch music.py "%command%"
