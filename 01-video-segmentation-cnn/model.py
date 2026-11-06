"""Multi-Stage Temporal Convolutional Network (MS-TCN++) for Video Action Segmentation.

This module implements dilated residual convolutions for fine-grained temporal action
and object segmentation across video sequences.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):
    """Dilated 1D residual convolutional block with exponential receptive field expansion."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual addition."""
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class SingleStageModel(nn.Module):
    """Single stage TCN with stacked dilated residual layers."""

    def __init__(self, num_layers: int, num_f_maps: int, in_dim: int, num_classes: int):
        super().__init__()
        self.conv_in = nn.Conv1d(in_dim, num_f_maps, kernel_size=1)
        self.layers = nn.ModuleList(
            [DilatedResidualLayer(num_f_maps, 2**i) for i in range(num_layers)]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process temporal feature sequence.

        Args:
            x: Input tensor of shape (batch_size, in_dim, sequence_length)

        Returns:
            Logits of shape (batch_size, num_classes, sequence_length)
        """
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MultiStageModel(nn.Module):
    """Multi-Stage TCN (MS-TCN++) architecture for iterative segmentation refinement."""

    def __init__(
        self,
        num_stages: int = 4,
        num_layers: int = 10,
        num_f_maps: int = 64,
        in_dim: int = 256,
        num_classes: int = 8,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.stage1 = SingleStageModel(num_layers, num_f_maps, in_dim, num_classes)
        self.stages = nn.ModuleList(
            [
                SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)
                for _ in range(num_stages - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Compute progressive segmentation predictions across all stages.

        Args:
            x: Input video feature tensor (batch_size, in_dim, seq_len)

        Returns:
            List of stage predictions, each shape (batch_size, num_classes, seq_len)
        """
        outputs: List[torch.Tensor] = []
        out = self.stage1(x)
        outputs.append(out)

        for stage in self.stages:
            prob = F.softmax(out, dim=1)
            out = stage(prob)
            outputs.append(out)

        return outputs


class TemporalSmoothingLoss(nn.Module):
    """Truncated Mean Squared Error over temporal difference for smooth segmentation."""

    def __init__(self, threshold: float = 4.0):
        super().__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculates temporal smoothing loss between adjacent frames."""
        log_prob = F.log_softmax(logits, dim=1)
        diff = self.mse(log_prob[:, :, 1:], log_prob[:, :, :-1])
        clamped_diff = torch.clamp(diff, min=0.0, max=self.threshold)
        return torch.mean(clamped_diff)
