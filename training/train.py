"""Train the evaluation network on synthetic positions."""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import torch
from torch import nn

from chess_engine.model import SimpleEvaluator
from training.dataset import create_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the chess evaluation model.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on.")
    parser.add_argument("--save-path", type=str, default="models/best_model.pth", help="Checkpoint path.")
    parser.add_argument("--dataset-size", type=int, default=2000, help="Number of synthetic samples.")
    parser.add_argument("--max-plies", type=int, default=30, help="Max random plies when generating positions.")
    return parser.parse_args()


def train_one_epoch(
    model: SimpleEvaluator, loader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: SimpleEvaluator, loader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * features.size(0)
    return total_loss / len(loader.dataset)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    train_loader, val_loader = create_dataloaders(
        size=args.dataset_size, batch_size=args.batch_size, max_plies=args.max_plies, device=device
    )

    model = SimpleEvaluator().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved new best model to {args.save_path}")


if __name__ == "__main__":
    main()

