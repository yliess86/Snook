import torch
import torch.nn as nn

from snook.config import AttributeDict
from snook.data.dataset.label import LabelDataset
from snook.model.network.modules import ConvBN2d
from snook.model.network.modules import Mish


class LabelNet(nn.Module):
    def __init__(self, scale: float = 1.0, activation: nn.Module = Mish) -> None:
        super(LabelNet, self).__init__()
        scaled = lambda x: int(scale * x)
        n_class = len(LabelDataset.NAME2LABEL)

        self.classifier = nn.Sequential()
        self.classifier.add_module(    "convbn_1", ConvBN2d(3, scaled(16), 3, stride=2))
        self.classifier.add_module("activation_1", activation())
        self.classifier.add_module(    "convbn_2", ConvBN2d(scaled(16), scaled(24), 3, stride=2))
        self.classifier.add_module("activation_2", activation())
        self.classifier.add_module(    "convbn_3", ConvBN2d(scaled(24), scaled(32), 3, stride=2))
        self.classifier.add_module("activation_3", activation())
        self.classifier.add_module(    "convbn_4", nn.Conv2d(scaled(32), scaled(64), 3, stride=1))
        self.classifier.add_module("activation_4", activation())
        self.classifier.add_module(    "convbn_5", nn.Conv2d(scaled(64), n_class, 1, stride=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.classifier(x)
        out = torch.reshape(out, (out.size(0), out.size(1)))
        out = torch.log_softmax(out, dim=1)

        return out

    def fuse(self) -> None:
        for name, module in self.named_modules():
            if type(module) == ConvBN2d:
                fused_module = torch.quantization.fuse_modules(module, ["conv", "bn"], inplace=False)[0]
                names = name.split(".")
                root = self
                for name in names[:-1]:
                    root = root.__getattr__(name)
                root.__setattr__(names[-1], fused_module)

    @classmethod
    def from_config(cls, conf: AttributeDict) -> "LabelNet":
        activation = nn.ReLU if conf.activation == "ReLU" else nn.ReLU6 if conf.activation == "ReLU6" else Mish
        return cls(conf.scale, activation)