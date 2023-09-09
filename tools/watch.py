import argparse
import asyncio
import pathlib

import psutil


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watches over files for changes and runs shell command when they occur"
    )
    parser.add_argument("file", type=pathlib.Path, help="watch specific file")
    parser.add_argument("exec", help="command to execute on changes")
    parser.add_argument("-d", "--delay", default=0.5, type=float, help="file updates debounce delay in seconds")
    args = parser.parse_args()

    delay: float = args.delay
    exec: str = args.exec
    file: pathlib.Path = args.file

    while True:
        task = asyncio.create_task(asyncio.create_subprocess_shell(exec))
        mtime = file.stat().st_mtime

        while file.stat().st_mtime == mtime:
            await asyncio.sleep(delay)

        try:
            # См. https://stackoverflow.com/questions/4789837/
            process = psutil.Process(task.result().pid)
            for it in process.children(recursive=True):
                it.kill()
            process.kill()
        except psutil.NoSuchProcess:
            pass


if __name__ == "__main__":
    asyncio.run(main())
