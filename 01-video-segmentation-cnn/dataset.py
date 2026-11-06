"""Dataset loader and temporal sequence generator for video segmentation tasks."""

from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class VideoActionDataset(Dataset):
    """PyTorch Dataset for temporal video sequence modeling.

    Yields sliding window feature clips and corresponding frame labels.
    """

    def __init__(
        self,
        features: Optional[List[np.ndarray]] = None,
        labels: Optional[List[np.ndarray]] = None,
        seq_len: int = 120,
        in_dim: int = 256,
        num_classes: int = 8,
        num_samples: int = 100,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_dim = in_dim
        self.num_classes = num_classes

        if features is not None and labels is not None:
            self.features = features
            self.labels = labels
        else:
            # Generate reproducible structured synthetic sequences for demonstration
            rng = np.random.default_rng(seed=42)
            self.features = []
            self.labels = []
            for _ in range(num_samples):
                # Simulated feature trajectory with smooth temporal autocorrelation
                base = rng.standard_normal((in_dim, seq_len)).astype(np.float32)
                # Apply moving average filter to simulate coherent video flow
                for c in range(in_dim):
                    base[c] = np.convolve(base[c], np.ones(5) / 5.0, mode="same")

                # Sequence of action intervals
                seq_labels = np.zeros(seq_len, dtype=np.int64)
                segment_points = np.sort(rng.choice(range(10, seq_len - 10), size=num_classes - 1, replace=False))
                cur_cls = 0
                prev = 0
                for pt in segment_points:
                    seq_labels[prev:pt] = cur_cls
                    prev = pt
                    cur_cls += 1
                seq_labels[prev:] = cur_cls

                self.features.append(base)
                self.labels.append(seq_labels)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = torch.from_numpy(self.features[idx])  # (in_dim, seq_len)
        lbl = torch.from_numpy(self.labels[idx])     # (seq_len,)
        return feat, lbl


def get_data_loaders(
    batch_size: int = 8,
    seq_len: int = 120,
    in_dim: int = 256,
    num_classes: int = 8,
) -> Tuple[DataLoader, DataLoader]:
    """Factory creating training and validation data loaders."""
    train_ds = VideoActionDataset(
        seq_len=seq_len, in_dim=in_dim, num_classes=num_classes, num_samples=160
    )
    val_ds = VideoActionDataset(
        seq_len=seq_len, in_dim=in_dim, num_classes=num_classes, num_samples=40
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
