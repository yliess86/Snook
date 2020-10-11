import torch
import torch.nn as nn


class L2Loss(nn.Module):
    def __init__(self) -> None:
        super(L2Loss, self).__init__()

    def forward(self, _y: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        return torch.mean((_y - y) ** 2) if mask is None else torch.mean(mask * (_y - y) ** 2)


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega: float = 14, theta: float = 0.5, eps: float = 1, alpha: float = 2.1) -> None:
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.eps = eps
        self.alpha = alpha

    def forward(self, _y: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        delta_y = torch.abs(_y - y) if mask is None else mask * torch.abs(_y - y)
        
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        
        loss1 = self.omega * torch.log(1 + (delta_y1 / self.omega) ** (self.alpha - y1))

        A = self.omega * (1 / (1 + (self.theta / self.eps) ** (self.alpha - y2))) * (self.alpha - y2) * ((self.theta / self.eps) ** (self.alpha - y2 - 1)) * (1 / self.eps)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.eps, self.alpha - y2))
        loss2 = A * delta_y2 - C

        return (torch.sum(loss1) + torch.sum(loss2)) / (len(loss1) + len(loss2))