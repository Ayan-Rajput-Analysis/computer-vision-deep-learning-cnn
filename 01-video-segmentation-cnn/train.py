"""Training script for Multi-Stage Temporal Convolutional Network (MS-TCN++)."""

import argparse
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from model import MultiStageModel, TemporalSmoothingLoss
from dataset import get_data_loaders


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes overall frame-wise accuracy.

    Args:
        predictions: Predicted class indices (batch_size, seq_len)
        targets: Ground truth class indices (batch_size, seq_len)
    """
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return (correct / total) * 100.0


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    ce_loss_fn: nn.Module,
    smooth_loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    smooth_weight: float = 0.15,
) -> Dict[str, float]:
    """Execute one training epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        stage_outputs = model(features)

        loss = 0.0
        for out in stage_outputs:
            ce_loss = ce_loss_fn(out, targets)
            smooth_loss = smooth_loss_fn(out)
            loss += ce_loss + smooth_weight * smooth_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        final_preds = torch.argmax(stage_outputs[-1], dim=1)
        acc = compute_metrics(final_preds, targets)

        total_loss += loss.item()
        total_acc += acc

    num_batches = len(loader)
    return {"loss": total_loss / num_batches, "accuracy": total_acc / num_batches}


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model accuracy on validation set."""
    model.eval()
    total_acc = 0.0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            stage_outputs = model(features)
            final_preds = torch.argmax(stage_outputs[-1], dim=1)
            total_acc += compute_metrics(final_preds, targets)

    return total_acc / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train MS-TCN Video Action Segmentation")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MS-TCN++] Target device: {device}")

    model = MultiStageModel(
        num_stages=3, num_layers=8, num_f_maps=64, in_dim=256, num_classes=8
    ).to(device)

    train_loader, val_loader = get_data_loaders(batch_size=args.batch_size)
    ce_loss_fn = nn.CrossEntropyLoss()
    smooth_loss_fn = TemporalSmoothingLoss(threshold=4.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print("Beginning training loop...")
    for epoch in range(1, args.epochs + 1):
        train_res = train_epoch(
            model, train_loader, ce_loss_fn, smooth_loss_fn, optimizer, device
        )
        val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Loss: {train_res['loss']:.4f} | "
            f"Train Acc: {train_res['accuracy']:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )


if __name__ == "__main__":
    main()
