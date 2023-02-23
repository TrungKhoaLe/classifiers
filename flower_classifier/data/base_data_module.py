from pathlib import Path
import pytorch_lightning as pl
import argparse
from typing import Tuple, Collection, Union, Dict
import os
import torch
from torch.utils.data import ConcatDataset, DataLoader
from flower_classifier.data.util import BaseDataset
import flower_classifier.metadata.shared as metadata
from flower_classifier import util


def load_and_print_info(data_module_class):
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset.data_train.samples[1])


def _download_raw_dataset(metadata: Dict, dl_dirname: Path) -> Path:
    dl_dirname.mkdir(parents=True, exist_ok=True)
    filename = dl_dirname / metadata["filename"]

    if filename.exists():
        return filename
    print(f"Downloading raw dataset from {metadata['url']} to {filename}...")
    util.download_url(metadata["url"], filename)
    return filename


# define default constants
BATCH_SIZE = 128
NUM_AVAIL_CPUS = len(os.sched_getaffinity(0))
NUM_AVAIL_GPUS = torch.cuda.device_count()
# sensible multiprocessing defaults: at most one worker per CPU
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS
# but in distributed data parallel mode, we launch a training on each GPU, so must divide out to keep total at one worker per CPU
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS // NUM_AVAIL_GPUS if NUM_AVAIL_GPUS else DEFAULT_NUM_WORKERS


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", DEFAULT_NUM_WORKERS)

        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # Make sure to set the variables below in subclasses
        self.input_dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        self.mapping: Collection
        self.data_train: Union[BaseDataset, ConcatDataset]
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]

    @classmethod
    def data_dirname(cls):
        return metadata.DATA_DIRNAME

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size",
            type=int,
            default=BATCH_SIZE,
            help=f"Number of examples to operate on per forward step. Default is {BATCH_SIZE}.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=DEFAULT_NUM_WORKERS,
            help=f"Number of additional processes to load data. Default is {DEFAULT_NUM_WORKERS}.",
        )
        return parser

    def config(self):
        return {"input_dims": self.input_dims, "output_dims": self.output_dims, "mapping": self.mapping}

    def prepare_data(self, *args, **kwargs):
        """
        Use this method to do things that might write to disk or that need to be done only from
        a single GPU in distributed settings (so don't set state `self.x = y`)
        """
        raise NotImplementedError

    def setup(self, stage: Union[str, None]):
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val,
        and optionally self.data_test
        """
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.batch_size, pin_memory=self.on_gpu)

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False, batch_size=self.batch_size, pin_memory=self.on_gpu)

    def predict_dataloader(self):
        """
        This method will be defined later. For example, streaming data to the
        service
        """
        pass
