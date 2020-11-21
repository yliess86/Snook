import torch
import torch.nn.functional as F


def peak_detection(heatmap: torch.Tensor) -> torch.Tensor:
    W, H = heatmap.size()
    
    augmented_heatmap = F.pad(heatmap, [1] * 4, mode="constant")
    center = augmented_heatmap[1:1 + W, 1:1 + H]
    left = augmented_heatmap[1:1 + W, 2:2 + H]
    right = augmented_heatmap[1:1 + W, 0:0 + H]
    up = augmented_heatmap[2:2 + W, 1:1 + H]
    down = augmented_heatmap[0:0 + W, 1:1 + H]
    
    peaks = torch.nonzero((
        (center > left) & (center > right) & (center > up) & (center > down)
    ), as_tuple=False).float()
    for xy in peaks:
        distance = torch.sqrt(((peaks - xy.unsqueeze(0)) ** 2).sum(dim=1))
        peaks = peaks[~((distance > 1) & (distance < 5))]
        
    return peaks.long()


def peak_windows(
    render: torch.Tensor, peaks: torch.Tensor, size: int = 64
) -> torch.Tensor:
    offset = size // 2
    augmented_render = F.pad(render, pad=[offset] * 4, mode="constant")
    return torch.cat([
        render[:, x - offset:x + offset, y - offset:y + offset].unsqueeze(0)
        for x, y in peaks
    ])