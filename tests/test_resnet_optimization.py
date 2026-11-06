"""Unit tests for SE-ResNet optimization architecture and augmentations."""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "02-deep-learning-optimization-resnet"))

try:
    import torch
    from model import SEResNet, SqueezeAndExcitation
    from augmentations import LabelSmoothingCrossEntropy, apply_cutmix, apply_mixup
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestSEResNetOptimization(unittest.TestCase):
    """Test suite for SE-ResNet architecture, attention blocks, and augmentations."""

    def test_se_attention_block(self):
        """Verify Squeeze-and-Excitation preserves spatial and channel dimensions."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not installed in local environment")
        b, c, h, w = 2, 32, 8, 8
        se = SqueezeAndExcitation(channels=c, reduction=8)
        x = torch.randn(b, c, h, w)
        out = se(x)
        self.assertEqual(out.shape, (b, c, h, w))

    def test_se_resnet_forward(self):
        """Verify SE-ResNet forward pass yields correct number of class logits."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not installed in local environment")
        model = SEResNet(layers=[1, 1, 1, 1], num_classes=5)
        x = torch.randn(2, 3, 32, 32)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 5))

    def test_label_smoothing_loss(self):
        """Verify label smoothing loss computation."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not installed in local environment")
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        preds = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])
        loss = criterion(preds, targets)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0.0)

    def test_cutmix_preserves_shape(self):
        """Verify CutMix keeps tensor shape intact."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not installed in local environment")
        x = torch.randn(4, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3])
        x_mixed, ta, tb, lam = apply_cutmix(x.clone(), y)
        self.assertEqual(x_mixed.shape, x.shape)
        self.assertTrue(0.0 <= lam <= 1.0)


if __name__ == "__main__":
    unittest.main()
