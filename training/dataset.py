"""PyTorch Dataset for chess positions."""

from __future__ import annotations

import random
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from chess_engine.evaluation import encode_game_state, simple_material_eval
from chess_engine.game_state import GameState
from chess_engine.move_generation import generate_legal_moves


def random_position(max_plies: int = 30) -> GameState:
    """Play random legal moves from the start position to create a noisy sample."""
    state = GameState.starting_state()
    plies = random.randint(0, max_plies)
    for _ in range(plies):
        moves = generate_legal_moves(state)
        if not moves:
            break
        state.make_move(random.choice(moves))
    return state


class PositionDataset(Dataset):
    """Dataset of encoded positions with material-based labels."""

    def __init__(self, size: int = 1000, max_plies: int = 30, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.samples: list[Tuple[torch.Tensor, float]] = []
        for _ in range(size):
            state = random_position(max_plies=max_plies)
            features = encode_game_state(state, device=device)
            label = torch.tensor(simple_material_eval(state), dtype=torch.float32, device=device)
            self.samples.append((features, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features, label = self.samples[idx]
        return features, label


def create_dataloaders(
    size: int = 1000,
    val_split: float = 0.2,
    batch_size: int = 32,
    max_plies: int = 30,
    device: torch.device | str = "cpu",
) -> Tuple[DataLoader, DataLoader]:
    dataset = PositionDataset(size=size, max_plies=max_plies, device=device)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

