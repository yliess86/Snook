import torch
import torch.nn as nn


class AdaptiveWingLoss(nn.Module):
    def __init__(
        self,
        omega: float = 14,
        theta: float = 0.5,
        eps: float = 1.0,
        alpha: float = 2.1,
    ) -> None:
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.eps = eps
        self.alpha = alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        delta = torch.abs(pred - target)
        if mask is not None: delta = mask * delta

        delta1 = delta[delta < self.theta]
        delta2 = delta[delta >= self.theta]

        pred1 = pred[delta < self.theta]
        pred2 = pred[delta >= self.theta]

        loss1 = self.omega * torch.log(
            1 + (delta1 / self.omega) ** (self.alpha - pred1)
        )

        A = (
            self.omega 
            * (1 / (1 + (self.theta / self.eps) ** (self.alpha - pred2)))
            * (self.alpha - pred2)
            * ((self.theta / self.eps) ** (self.alpha - pred2 - 1))
            * (1 / self.eps)
        )
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.eps, self.alpha - pred2)
        )
        loss2 = A * delta2 - C

        return (
            (torch.sum(loss1) + torch.sum(loss2)) / (len(loss1) + len(loss2))
        )