"""Detects a paragraph of text in an input image.

Example usage as a script:
  python flower_classifier/flower_predictor.py \
    flower_classifier/tests/images/image_04473.jpg

  python flower_classifier/flower_predictor.py \
    https://{private-domain}.s3-us-west-2.amazonaws.com/images/image_04473.jpg

"""
import argparse
from pathlib import Path
from typing import Sequence, Union

from PIL import Image
import torch

from flower_classifier import util
from flower_classifier.stems.image import FlowerStem

STAGED_MODEL_DIRNAME = Path(__file__).resolve().parent / "artifacts" / "flower_classifier"
MODEL_FILE = "model.pt"


class FlowerPredictor:
    """Recognizes what flower is in an image"""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = STAGED_MODEL_DIRNAME / MODEL_FILE
        self.model = torch.jit.load(model_path)
        self.mapping = self.model.mapping
        self.stem = FlowerStem()

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> str:
        """Predict/infer text in input image (which can be a file path or url)."""
        image_pil = image
        if not isinstance(image, Image.Image):
            image_pil = util.read_image_pil(image, grayscale=False)

        image_tensor = self.stem(image_pil).unsqueeze(axis=0)
        y_pred = torch.argmax(self.model(image_tensor)[0])
        pred_str = convert_y_label_to_string(y=y_pred, mapping=self.mapping)

        return pred_str


def convert_y_label_to_string(y: torch.Tensor, mapping: Sequence[str]) -> str:
    # The indices of mapping start from 0 , while the class indices start from 1
    y = y - 1
    return mapping[y]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        type=str,
        help="""Name for an image file. This can be a local path, a URL,
        a URI from AWS/GCP/Azure storage, an HDFS path, or any other resource locator
        supported by the smart_open library.""",
    )
    args = parser.parse_args()

    text_recognizer = FlowerPredictor()
    pred_str = text_recognizer.predict(args.filename)
    print(pred_str)


if __name__ == "__main__":
    main()
