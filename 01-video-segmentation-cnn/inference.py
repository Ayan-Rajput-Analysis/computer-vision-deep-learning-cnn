"""Real-time inference and segmentation visualization pipeline."""

from typing import List
import numpy as np
import torch
from model import MultiStageModel

CLASS_NAMES = [
    "Background",
    "Preparation",
    "Assembly",
    "Tool Manipulation",
    "Quality Inspection",
    "Fastening",
    "Disassembly",
    "Packaging",
]

CLASS_COLORS = [
    (128, 128, 128),
    (255, 99, 71),
    (50, 205, 50),
    (30, 144, 255),
    (255, 215, 0),
    (138, 43, 226),
    (255, 20, 147),
    (0, 255, 255),
]


def predict_sequence(
    model: MultiStageModel, feature_sequence: torch.Tensor, device: torch.device
) -> np.ndarray:
    """Predict action classes across a temporal feature sequence.

    Args:
        model: Trained MS-TCN model
        feature_sequence: Tensor of shape (in_dim, seq_len)
        device: PyTorch device

    Returns:
        1D numpy array of predicted class indices of length seq_len
    """
    model.eval()
    with torch.no_grad():
        x = feature_sequence.unsqueeze(0).to(device)  # (1, in_dim, seq_len)
        stage_outputs = model(x)
        final_logits = stage_outputs[-1]  # (1, num_classes, seq_len)
        predictions = torch.argmax(final_logits, dim=1).squeeze(0)
    return predictions.cpu().numpy()


def render_timeline_ascii(predictions: np.ndarray) -> str:
    """Render an ASCII timeline representation of segmented action intervals."""
    intervals = []
    if len(predictions) == 0:
        return "Empty timeline."

    cur_cls = predictions[0]
    start = 0
    for idx, cls in enumerate(predictions):
        if cls != cur_cls:
            intervals.append((start, idx - 1, cur_cls))
            start = idx
            cur_cls = cls
    intervals.append((start, len(predictions) - 1, cur_cls))

    lines = ["Action Segmentation Timeline:", "=" * 60]
    for start, end, cls in intervals:
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class_{cls}"
        duration = end - start + 1
        lines.append(f"Frames [{start:04d} - {end:04d}] ({duration:03d} frames) -> {name}")
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    device = torch.device("cpu")
    model = MultiStageModel(
        num_stages=3, num_layers=6, num_f_maps=32, in_dim=128, num_classes=len(CLASS_NAMES)
    ).to(device)

    # Test sample with 150 frames
    sample_feat = torch.randn(128, 150)
    preds = predict_sequence(model, sample_feat, device)
    print(render_timeline_ascii(preds))
