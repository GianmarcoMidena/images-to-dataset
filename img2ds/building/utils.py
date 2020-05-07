from pathlib import Path


def iter_folders(folder: Path):
    for path in folder.iterdir():
        if path.is_dir():
            yield path


def iter_files(folder: Path):
    for path in folder.iterdir():
        if path.is_file():
            yield path
