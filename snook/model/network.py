from typing import Tuple, Union

import torch
import torch.nn as nn


EncoderProjection = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


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
        activation: nn.Module = nn.ReLU
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
        activation: nn.Module = nn.ReLU,
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
        activation: nn.Module = nn.ReLU,
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