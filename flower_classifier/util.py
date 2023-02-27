from typing import Union
from pathlib import Path
import contextlib
import os
import gdown
from PIL import Image
import smart_open
from io import BytesIO
import base64


def download_url(url, filename):
    gdown.download(url, str(filename))


def read_image_pil(image_uri: Union[Path, str], grayscale=False) -> Image:
    with smart_open.open(image_uri, "rb") as image_file:
        return read_image_pil_file(image_file, grayscale)


def read_image_pil_file(image_file, grayscale=False) -> Image:
    with Image.open(image_file) as image:
        if grayscale:
            image = image.convert(mode="L")
        else:
            image = image.convert(mode=image.mode)
        return image


def encode_b64_image(image, format="png"):
    """Encode a PIL image as a base64 string."""
    _buffer = BytesIO()  # bytes that live in memory
    image.save(_buffer, format=format)  # but which we write to like a file
    encoded_image = base64.b64encode(_buffer.getvalue()).decode("utf8")
    return encoded_image


@contextlib.contextmanager
def temporary_working_directory(working_dir: Union[str, Path]):
    """Temporarily switches to a directory, then returns to the original directory on exit."""
    curdir = os.getcwd()
    os.chdir(working_dir)
    try:
        yield
    finally:
        os.chdir(curdir)
