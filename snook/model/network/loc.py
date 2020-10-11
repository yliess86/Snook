import torch
import torch.nn as nn

from snook.config import AttributeDict
from snook.model.network.modules import ConvBN2d
from snook.model.network.modules import InvertedResidual
from snook.model.network.modules import Mish
from typing import List
from typing import Tuple
from typing import Union


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t: int, repeat: int, activation: nn.Module, projection: bool = True) -> None:
        super(EncoderBlock, self).__init__()
        self.transition = InvertedResidual(in_channels, out_channels, stride=2, t=t, residual=False, activation=activation)
        self.block = nn.Sequential()
        for i in range(repeat):
            self.block.add_module(f"invres_{i}", InvertedResidual(out_channels, out_channels, t=t, residual=True, activation=activation))
        self.projection = ConvBN2d(out_channels, out_channels, 1, bias=False) if projection else None

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out = self.block(self.transition(x))
        if self.projection is not None:
            return out, self.projection(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t: int, repeat: int, activation: nn.Module, scale: int = 2) -> None:
        super(DecoderBlock, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale)
        self.transition = InvertedResidual(in_channels, out_channels, stride=1, t=t, residual=False, activation=activation)
        self.block = nn.Sequential()
        for i in range(repeat):
            self.block.add_module(f"invres_{i}", InvertedResidual(out_channels, out_channels, t=t, residual=True, activation=activation))

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.block(self.transition(self.upsample(x) + residual))


class LocNet(nn.Module):
    def __init__(self, layers: List[Tuple[int, int, int]], scale: float = 1.0, repeat: int = 2, activation: nn.Module = Mish) -> None:
        super(LocNet, self).__init__()
        scaled = lambda x: int(scale * x)
        first, last = layers[0].inp, layers[-1].out

        self.preprocess = nn.Sequential()
        self.preprocess.add_module(    "convbn_1", ConvBN2d(3, scaled(first), 3, stride=2, padding=1, bias=False))
        self.preprocess.add_module("activation_1", activation())
        self.preprocess.add_module(    "convbn_2", ConvBN2d(scaled(first), scaled(first), 3, padding=1, bias=False))
        self.preprocess.add_module("activation_2", activation())

        self.encoders = nn.ModuleList([EncoderBlock(scaled(l.inp), scaled(l.out), l.t, repeat, activation) for l in layers])
        self.bottleneck = EncoderBlock(scaled(last), scaled(last), 6, repeat, activation, projection=False)
        self.decoders = nn.ModuleList([DecoderBlock(scaled(l.out), scaled(l.inp), l.t, repeat, activation, 2) for l in layers[::-1]])

        self.postprocess = nn.Sequential()
        self.postprocess.add_module(  "upsample_1", nn.UpsamplingBilinear2d(scale_factor=2))
        self.postprocess.add_module(    "convbn_1", ConvBN2d(scaled(first), scaled(first), 3, padding=1, bias=False))
        self.postprocess.add_module("activation_1", activation())
        self.postprocess.add_module(  "upsample_2", nn.UpsamplingBilinear2d(scale_factor=2))

        self.out = nn.Conv2d(scaled(first), 1, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.preprocess(x)
        
        skips = []
        for encoder in self.encoders:
            out, skip = encoder(out)
            skips.append(skip)

        out = self.bottleneck(out)

        for i, decoder in enumerate(self.decoders):
            skip = skips[len(skips) - i - 1]
            out = decoder(out, skip)

        out = self.postprocess(out)
        out = self.out(out).squeeze(1)

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
    def from_config(cls, conf: AttributeDict) -> "LocNet":
        activation = nn.ReLU if conf.activation == "ReLU" else nn.ReLU6 if conf.activation == "ReLU6" else Mish
        return cls(conf.layers, conf.scale, conf.repeat, activation)