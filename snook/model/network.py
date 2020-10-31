from typing import List, NamedTuple, Tuple, Union

import torch
import torch.nn as nn


EncoderProjection = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class Layer(NamedTuple):
    inp: int
    out: int
    t: int


class ConvBn2d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        groups: int = 1,
    ) -> None:
        super(ConvBn2d, self).__init__()
        self.add_module("conv", nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups
        ))
        self.add_module("bn", nn.BatchNorm2d(out_channels))


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        t: int = 6,
        residual: bool = True,
        activation: nn.Module = nn.ReLU, # type: ignore
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.residual = residual
        
        self.expension = nn.Sequential()
        self.expension.add_module("convbn", ConvBn2d(
            in_planes, in_planes * t, kernel_size=1, bias=False
        ))
        self.expension.add_module("activation", activation())
        
        self.depthwise = nn.Sequential()
        self.depthwise.add_module("convbn", ConvBn2d(
            in_planes * t,
            in_planes * t,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=in_planes * t
        ))
        self.depthwise.add_module("activation", activation())

        self.projection = ConvBn2d(
            in_planes * t, out_planes, kernel_size=1, bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.projection(self.depthwise(self.expension(x)))
        if self.residual: out += x
        return out


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        t: int,
        repeat: int = 1,
        activation: nn.Module = nn.ReLU, # type: ignore
        projection: bool = True
    ) -> None:
        super(EncoderBlock, self).__init__()
        self.transition = InvertedResidual(
            in_channels,
            out_channels,
            stride=2,
            t=t,
            residual=False,
            activation=activation
        )
        
        self.block = nn.Sequential()
        for i in range(repeat):
            self.block.add_module(f"invres_{i}", InvertedResidual(
                out_channels,
                out_channels,
                t=t,
                residual=True,
                activation=activation
            ))

        self.projection = ConvBn2d(
            out_channels,
            out_channels,
            kernel_size=1,
            bias=False
        ) if projection else None

    def forward(self, x: torch.Tensor) -> EncoderProjection:
        out = self.block(self.transition(x))
        if self.projection is not None: return out, self.projection(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        t: int,
        repeat: int = 1,
        activation: nn.Module = nn.ReLU, # type: ignore
        scale: int = 2
    ) -> None:
        super(DecoderBlock, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale)
        self.transition = InvertedResidual(
            in_channels,
            out_channels,
            stride=1,
            t=t,
            residual=False,
            activation=activation
        )

        self.block = nn.Sequential()
        for i in range(repeat):
            self.block.add_module(f"invres_{i}", InvertedResidual(
                out_channels,
                out_channels,
                t=t,
                residual=True,
                activation=activation
            ))

    def forward(
        self, x: torch.Tensor, *, residual: torch.Tensor
    ) -> torch.Tensor:
        return self.block(self.transition(self.upsample(x) + residual))


class AutoEncoder(nn.Module):
    def __init__(
        self,
        layers: List[Layer],
        in_channels: int,
        out_channels: int,
        scale: float = 1.0,
        repeat: int = 2,
        activation: nn.Module = nn.ReLU, # type: ignore
    ) -> None:
        super(AutoEncoder, self).__init__()
        scaled = lambda x: int(scale * x)
        first, last = layers[0].inp, layers[-1].out

        self.preprocess = nn.Sequential()
        self.preprocess.add_module("convbn_1", ConvBn2d(
            in_channels,
            scaled(first),
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        ))
        self.preprocess.add_module("activation_1", activation())
        self.preprocess.add_module("convbn_2", ConvBn2d(
            scaled(first), scaled(first), kernel_size=3, padding=1, bias=False
        ))
        self.preprocess.add_module("activation_2", activation())

        self.encoders = nn.ModuleList([
            EncoderBlock(
                scaled(layer.inp),
                scaled(layer.out),
                t=layer.t,
                repeat=repeat,
                activation=activation,
            )
            for layer in layers
        ])
        self.bottleneck = EncoderBlock(
            scaled(last),
            scaled(last),
            t=6,
            repeat=repeat,
            activation=activation,
            projection=False,
        )
        self.decoders = nn.ModuleList([
            DecoderBlock(
                scaled(layer.out),
                scaled(layer.inp),
                t=layer.t,
                repeat=repeat,
                activation=activation,
                scale=2,
            )
            for layer in layers[::-1]
        ])

        self.postprocess = nn.Sequential()
        self.postprocess.add_module("upsample_1", nn.UpsamplingBilinear2d(
            scale_factor=2
        ))
        self.postprocess.add_module("convbn_1", ConvBn2d(
            scaled(first), scaled(first), kernel_size=3, padding=1, bias=False
        ))
        self.postprocess.add_module("activation_1", activation())
        self.postprocess.add_module("upsample_2", nn.UpsamplingBilinear2d(
            scale_factor=2
        ))

        self.out = nn.Conv2d(
            scaled(first), out_channels, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.preprocess(x)
        
        skips = []
        for encoder in self.encoders:
            out, skip = encoder(out)
            skips.append(skip)

        out = self.bottleneck(out)

        for i, decoder in enumerate(self.decoders):
            skip = skips[len(skips) - i - 1]
            out = decoder(out, residual=skip)

        out = self.postprocess(out)
        out = self.out(out)

        return out

    def fuse(self) -> None:
        for name, module in self.named_modules():
            if type(module) == ConvBn2d:
                fused_module = torch.quantization.fuse_modules(
                    module, ["conv", "bn"], inplace=False
                )[0]

                names = name.split(".")
                root = self
                for name in names[:-1]:
                    root = root.__getattr__(name) # type: ignore
                
                root.__setattr__(names[-1], fused_module)


class Classifier(nn.Module):
    def __init__(
        self,
        layers: List[Layer],
        *,
        hidden: int,
        n_class: int,
        activation: nn.Module = nn.ReLU, # type: ignore
    ) -> None:
        super(Classifier, self).__init__()
        self.features = nn.Sequential(*(
            nn.Sequential(
                ConvBn2d(layer.inp, layer.out, kernel_size=layer.t, stride=2),
                activation(),
            )
            for layer in layers
        ))

        self.classifier = nn.Sequential(
            nn.Conv2d(layers[-1].out, hidden, kernel_size=1),
            nn.Dropout(),
            activation(),
            nn.Conv2d(hidden, n_class, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).view((x.size(0), -1))