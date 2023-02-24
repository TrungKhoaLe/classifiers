from torchvision import models
from colections import OrderedDict
from typing import Dict, Any
from torch import nn
import argparse
import torch


# default constants
FC1_DIM = 4096
FC2_DIM = 4096
FC_DROPOUT = 0.2


class VGG16Classifier(nn.Module):
    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config
        input_channels, input_height, input_width = self.data_config["input_dims"]
        assert (
            input_height == input_width
        ), f"input height and width should be equal, but was {input_height}, {input_width}"
        self.input_height, self.input_width = input_height, input_width
        num_classes = len(self.data_config["mapping"])

        fc1_dim = self.args.get("fc1_dim", FC1_DIM)
        fc2_dim = self.args.get("fc2_dim", FC2_DIM)
        fc_dropout = self.args.get("fc_dropout", FC_DROPOUT)

        backbone = models.vgg16(pretrained=True)
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        fc_input_dim = backbone.classifier[0].in_features
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(fc_input_dim, fc1_dim)),
                    ("relu1", nn.ReLU()),
                    ("dropout1", nn.Dropout(fc_dropout)),
                    ("fc2", nn.Linear(fc1_dim, fc2_dim)),
                    ("relu2", nn.ReLU()),
                    ("dropout2", nn.Dropout(fc_dropout)),
                    ("fc3", nn.Linear(fc2_dim, num_classes)),
                ]
            )
        )

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--fc1_dim", type=int, default=FC1_DIM)
        parser.add_argument("--fc2_dim", type=int, default=FC2_DIM)
        parser.add_argument("--fc_dropout", type=float, default=FC_DROPOUT)
        return parser
