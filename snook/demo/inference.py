import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snook.demo.objects import Ball
from snook.demo.objects import Cue
from snook.demo.objects import Debug
from snook.demo.objects import Pool
from typing import List
from typing import Tuple


class Snook(nn.Module):
    def __init__(self, loc: str, mask: str, label: str, dir: str) -> None:
        super(Snook, self).__init__()
        self.loc   = torch.jit.load(loc)
        self.mask  = torch.jit.load(mask)
        self.label = torch.jit.load(label)
        self.dir   = torch.jit.load(dir)

    def clamp_alpha(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        x = torch.clamp(x, 0, 1)
        x[x < alpha] = 0
        return x        

    def peak_detection(self, x: torch.Tensor) -> torch.Tensor:
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
            dist = torch.sqrt(torch.sum((coords.float() - pos.unsqueeze(0)) ** 2, dim=1))
            reject = (dist < 5) & (dist > 1)
            coords = coords[~reject]

        return coords

    def build_windows(self, x: torch.Tensor, peaks: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, pad=(16, 16, 16, 16), mode="constant")
        windows = []
        for peak in peaks:
            window = x[:, :, peak[0]:peak[0] + 32, peak[1]:peak[1] + 32]
            windows.append(window)
        windows = torch.cat(windows)
        return windows

    def n_items(self, labels: torch.Tensor, label: int) -> int:
        pass

    def forward(self, x: torch.Tensor, alphas: List[float]) -> Pool:
        heatmap_alpha, mask_alpha, logits_alpha = alphas

        h_x = F.interpolate(x, size=(256, 256), mode="bilinear")
        m_x = F.interpolate(x, size=( 64,  64), mode="bilinear")

        heatmap = self.clamp_alpha(self.loc (h_x), heatmap_alpha).unsqueeze(0)
        mask = self.clamp_alpha(self.mask(m_x), mask_alpha).unsqueeze(0)
        heatmap = (heatmap * F.interpolate(mask, size=(256, 256)))[0, 0]
        peaks = self.peak_detection(heatmap)

        if len(peaks) <= 0:
            return Pool()
            
        windows = self.build_windows(h_x, peaks)
        logits = torch.exp(self.label(windows))
        logits, labels = torch.max(logits, dim=-1)

        valid = logits > logits_alpha
        peaks, windows = peaks[valid], windows[valid]
        logits, labels = logits[valid], labels[valid]

        balls, cues = [], []

        for i in range(4):
            select = labels == i
            if (n := select.long().sum().item()) > 0:
                idxs = torch.topk(logits[select], k=np.min([1 if i < 2 else 6, n]))[-1]
                peak = peaks[select][idxs].cpu().numpy()
                balls += [Ball(p, i) for p in peak]

        select = labels == 4
        if (n := select.long().sum().item()) > 0:
            _, idxs = torch.topk(logits[select], k=np.min([2, n]))
            peak    = peaks[select][idxs].cpu().numpy()
            dir     = self.dir(windows[select][idxs]).cpu().numpy()[::-1]
            cues    = [Cue(p, d) for p, d in zip(peak, dir)]

        debug = Debug(
            h_x.squeeze(0).permute((1, 2, 0)).cpu().numpy(),
            mask.squeeze(0).permute((1, 2, 0)).repeat(1, 1, 3).cpu().numpy(),
            heatmap.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy(),
        )

        return Pool(balls, cues, debug)