"""Unit tests for MS-TCN++ video segmentation model."""

import unittest
import sys
from pathlib import Path

# Add project root and subdirectories to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "01-video-segmentation-cnn"))

try:
    import torch
    from model import MultiStageModel, SingleStageModel, TemporalSmoothingLoss
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestVideoSegmentation(unittest.TestCase):
    """Test suite for MS-TCN++ layers and losses."""

    def test_single_stage_tcn_shape(self):
        """Verify single-stage TCN output dimensions match input temporal length."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not installed in local environment")
        batch_size = 2
        in_dim = 64
        num_classes = 5
        seq_len = 30
        model = SingleStageModel(num_layers=4, num_f_maps=32, in_dim=in_dim, num_classes=num_classes)
        x = torch.randn(batch_size, in_dim, seq_len)
        out = model(x)
        self.assertEqual(out.shape, (batch_size, num_classes, seq_len))

    def test_multi_stage_tcn_stages(self):
        """Verify multi-stage outputs return correct number of stages and shapes."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not installed in local environment")
        num_stages = 3
        num_classes = 6
        seq_len = 25
        model = MultiStageModel(num_stages=num_stages, num_layers=4, num_f_maps=16, in_dim=32, num_classes=num_classes)
        x = torch.randn(2, 32, seq_len)
        outputs = model(x)
        self.assertEqual(len(outputs), num_stages)
        for out in outputs:
            self.assertEqual(out.shape, (2, num_classes, seq_len))

    def test_temporal_smoothing_loss(self):
        """Verify temporal smoothing loss computes non-negative scalar."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not installed in local environment")
        loss_fn = TemporalSmoothingLoss(threshold=4.0)
        logits = torch.randn(2, 4, 20)
        loss = loss_fn(logits)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
