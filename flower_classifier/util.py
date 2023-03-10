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


def read_b64_image(b64_string, grayscale=False):
    """Load base64-encoded images."""
    try:
        image_file = read_b64_string(b64_string)
        return read_image_pil_file(image_file, grayscale)
    except Exception as exception:
        raise ValueError("Could not load image from b64 {}: {}".format(b64_string, exception)) from exception


def read_b64_string(b64_string, return_data_type=False):
    """Read a base64-encoded string into an in-memory file-like object."""
    data_header, b64_data = split_and_validate_b64_string(b64_string)
    b64_buffer = BytesIO(base64.b64decode(b64_data))
    if return_data_type:
        return get_b64_filetype(data_header), b64_buffer
    else:
        return b64_buffer


def get_b64_filetype(data_header):
    """Retrieves the filetype information from the data type header of a base64-encoded object."""
    _, file_type = data_header.split("/")
    return file_type


def split_and_validate_b64_string(b64_string):
    """Return the data_type and data of a b64 string, with validation."""
    header, data = b64_string.split(",", 1)
    assert header.startswith("data:")
    assert header.endswith(";base64")
    data_type = header.split(";")[0].split(":")[1]
    return data_type, data


def encode_b64_image(image, format="jpg"):
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
