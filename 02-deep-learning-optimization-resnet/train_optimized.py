"""Optimized deep learning training loop with mixed precision and cosine scheduling."""

import argparse
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import SEResNet
from augmentations import LabelSmoothingCrossEntropy, apply_cutmix


def compute_accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> Tuple[float, ...]:
    """Computes precision@k for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return tuple(res)


def create_synthetic_vision_data(
    num_samples: int = 128, num_classes: int = 10, img_size: int = 64
) -> DataLoader:
    """Creates synthetic image dataset for pipeline validation."""
    torch.manual_seed(42)
    x = torch.randn(num_samples, 3, img_size, img_size)
    y = torch.randint(0, num_classes, (num_samples,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=16, shuffle=True)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_cutmix: bool = True,
) -> Tuple[float, float]:
    """Train for one epoch with optional CutMix augmentation."""
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if use_cutmix and torch.rand(1).item() > 0.5:
            x, target_a, target_b, lam = apply_cutmix(x, y)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(x)
                loss = lam * criterion(out, target_a) + (1.0 - lam) * criterion(out, target_b)
        else:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()

        top1, _ = compute_accuracy(out, y, topk=(1, 5))
        running_loss += loss.item()
        running_acc += top1

    return running_loss / len(loader), running_acc / len(loader)


def main():
    parser = argparse.ArgumentParser(description="SE-ResNet Accuracy Optimization")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SE-ResNet Optimization] Running on: {device}")

    # Use compact configuration for testing/demonstration
    model = SEResNet(layers=[2, 2, 2, 2], num_classes=10).to(device)
    train_loader = create_synthetic_vision_data(num_samples=96, num_classes=10)

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print("Executing optimization loop...")
    for epoch in range(1, args.epochs + 1):
        loss, acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        scheduler.step()
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] LR: {scheduler.get_last_lr()[0]:.6f} | Loss: {loss:.4f} | Top-1: {acc:.2f}%")


if __name__ == "__main__":
    main()
