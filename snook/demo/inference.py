import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class Snook(nn.Module):
    def __init__(self, loc: str, mask: str, label: str) -> None:
        super(Snook, self).__init__()
        self.loc   = torch.jit.load(loc)
        self.mask  = torch.jit.load(mask)
        self.label = torch.jit.load(label)

    def _heatmap(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        heatmap = torch.clamp(self.loc(x), 0, 1)
        heatmap[heatmap < alpha] = 0
        return heatmap

    def _mask(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        mask = torch.clamp(self.mask(x), 0, 1)
        mask[mask < alpha] = 0
        return mask        

    def _coordinates(self, x: torch.Tensor) -> torch.Tensor:
        W, H = x.size()

        border = F.pad(x, [1] * 4, mode="constant")
        center = border[1:1 + W, 1:1 + H]
        left   = border[1:1 + W, 2:2 + H]
        right  = border[1:1 + W, 0:0 + H]
        up     = border[2:2 + W, 1:1 + H]
        down   = border[0:0 + W, 1:1 + H]
        
        peaks  = (center > left) & (center > right) & (center > up) & (center > down)
        coords = torch.nonzero(peaks, as_tuple=False)
        for pos in coords:
            pos    = pos.unsqueeze(0)
            dist   = torch.sqrt(torch.sum((coords.float() - pos) ** 2, dim=1))
            reject = (dist < 3) & (dist > 1)
            coords = coords[~reject]

        return coords

    def _windows(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, pad=(16, 16, 16, 16), mode="constant")
        windows = []
        for coord in coords:
            window = x[:, :, coord[0]:coord[0] + 32, coord[1]:coord[1] + 32]
            windows.append(window)
        windows = torch.cat(windows)
        return windows

    def _classify(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        logits = torch.exp(self.label(x))
        labels = torch.argmax(logits, dim=-1)
        return labels, logits

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_x = F.interpolate(x, size=(256, 256), mode="bilinear")
        m_x = F.interpolate(x, size=( 64,  64), mode="bilinear")

        heatmap = self._heatmap(h_x, 0.4).unsqueeze(0)
        mask    = self._mask   (m_x, 0.6).unsqueeze(0)
        mask    = F.interpolate(mask, size=(256, 256), mode="nearest")
        heatmap = (heatmap * mask).squeeze(0).squeeze(0)
        
        coords  = self._coordinates(heatmap)
        if len(coords) <= 0:
            return heatmap, None, None, None

        windows = self._windows(h_x, coords)
        labels, logits = self._classify(windows)
        return heatmap, coords, logits, labels