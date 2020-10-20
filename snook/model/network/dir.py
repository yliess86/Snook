import torch
import torch.nn as nn
import torch.nn.functional as F

from snook.config import AttributeDict
from snook.model.network.modules import ConvBN2d
from snook.model.network.modules import Mish


class DirNet(nn.Module):
    def __init__(self, scale: float = 1.0, activation: nn.Module = Mish) -> None:
        super(DirNet, self).__init__()
        scaled = lambda x: int(scale * x)

        self.regressor = nn.Sequential()
        self.regressor.add_module(    "convbn_1", ConvBN2d(3, scaled(16), 3, stride=2))
        self.regressor.add_module("activation_1", activation())
        self.regressor.add_module(    "convbn_2", ConvBN2d(scaled(16), scaled(24), 3, stride=2))
        self.regressor.add_module("activation_2", activation())
        self.regressor.add_module(    "convbn_3", ConvBN2d(scaled(24), scaled(32), 3, stride=2))
        self.regressor.add_module("activation_3", activation())
        self.regressor.add_module(    "convbn_4", nn.Conv2d(scaled(32), scaled(64), 3, stride=1))
        self.regressor.add_module("activation_4", activation())
        self.regressor.add_module(    "convbn_5", nn.Conv2d(scaled(64), 2, 1, stride=1))

    def forward(self, x: torch.Tensor, eps: float = 1e-11) -> torch.Tensor:
        out = self.regressor(x)
        out = torch.reshape(out, (out.size(0), out.size(1)))
        mag = torch.sqrt(torch.sum(out * out, dim=-1, keepdim=True))
        out = out / (mag + eps)

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
    def from_config(cls, conf: AttributeDict) -> "DirNet":
        activation = nn.ReLU if conf.activation == "ReLU" else nn.ReLU6 if conf.activation == "ReLU6" else Mish
        return cls(conf.scale, activation)