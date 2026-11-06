"""Advanced data augmentations: CutMix, MixUp, and Label Smoothing."""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    """Generate random bounding box coordinates for CutMix."""
    w = size[2]
    h = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    # Uniform center point
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2


def apply_cutmix(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Applies CutMix augmentation to input batch."""
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size(0))

    target_a = y
    target_b = y[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to exact box area proportion
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, target_a, target_b, lam


def apply_mixup(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Applies linear MixUp augmentation between paired samples."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    rand_index = torch.randperm(x.size(0))

    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a = y
    target_b = y[rand_index]
    return mixed_x, target_a, target_b, lam


class LabelSmoothingCrossEntropy(nn.Module):
    """CrossEntropyLoss with uniform label smoothing regularization."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target, reduction="mean")
        return (1.0 - self.smoothing) * nll + (self.smoothing / num_classes) * loss
