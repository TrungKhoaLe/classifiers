from flower_classifier.data.base_data_module import _download_raw_dataset
from flower_classifier.data.base_data_module import load_and_print_info
from flower_classifier.data.base_data_module import BaseDataModule
import argparse
import flower_classifier.metadata.flowers as metadata
from torchvision import datasets
from flower_classifier.stems.image import FlowerStem
from flower_classifier.util import temporary_working_directory
import toml
import zipfile
from torchvision import transforms


RAW_DATA_DIRNAME = metadata.RAW_DATA_DIRNAME
METADATA_FILENAME = metadata.METADATA_FILENAME
DL_DATA_DIRNAME = metadata.DL_DATA_DIRNAME


class Flowers(BaseDataModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.data_dir = DL_DATA_DIRNAME
        self.train_transform = FlowerStem()
        self.test_transform = FlowerStem()
        self.test_transform.pil_transforms.transforms.clear()
        self.test_transform.pil_transforms.transforms.extend(
            [
                transforms.Resize(255),
                transforms.CenterCrop(224),
            ]
        )
        self.input_dims = metadata.DIMS
        self.output_dims = metadata.OUTPUT_DIMS
        self.mapping = metadata.MAPPING

    def prepare_data(self, *args, **kwargs) -> None:
        metadata = toml.load(METADATA_FILENAME)
        _download_raw_dataset(metadata, RAW_DATA_DIRNAME)
        with temporary_working_directory(RAW_DATA_DIRNAME):
            with zipfile.ZipFile(metadata["filename"], "r") as zf:
                zf.extractall(DL_DATA_DIRNAME)

    def setup(self, stage=None) -> None:
        train_dir = self.data_dir / "train"
        eval_dir = self.data_dir / "valid"
        test_dir = self.data_dir / "test"
        self.data_train = datasets.ImageFolder(train_dir, transform=self.train_transform)
        self.data_val = datasets.ImageFolder(eval_dir, transform=self.test_transform)
        self.data_test = datasets.ImageFolder(test_dir, transform=self.test_transform)


if __name__ == "__main__":
    load_and_print_info(Flowers)
