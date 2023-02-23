from typing import Union
from pathlib import Path
import contextlib
import os
import gdown


def download_url(url, filename):
    gdown.download(url, str(filename))


@contextlib.contextmanager
def temporary_working_directory(working_dir: Union[str, Path]):
    """Temporarily switches to a directory, then returns to the original directory on exit."""
    curdir = os.getcwd()
    os.chdir(working_dir)
    try:
        yield
    finally:
        os.chdir(curdir)
