"""Feature extraction and model-backed evaluation."""

from __future__ import annotations

from typing import Optional, Any

from .game_state import GameState
from .utils import PIECE_VALUES, piece_color

PIECE_TO_INDEX = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}

INPUT_DIM = 12 * 64 + 5


def encode_game_state(state: GameState, device: str = "cpu"):
    """Encode game state into a flat tensor suitable for the model.

    Imports `torch` lazily so module import doesn't require `torch` to be installed
    when only material evaluation is used.
    """
    import torch

    dev = torch.device(device) if isinstance(device, str) else device
    vec = torch.zeros(INPUT_DIM, device=dev, dtype=torch.float32)
    for idx, piece in enumerate(state.board.squares):
        if piece == ".":
            continue
        plane = PIECE_TO_INDEX[piece]
        vec[plane * 64 + idx] = 1.0
    offset = 12 * 64
    vec[offset] = 1.0 if state.side_to_move == "w" else -1.0
    vec[offset + 1] = 1.0 if "K" in state.castling_rights else 0.0
    vec[offset + 2] = 1.0 if "Q" in state.castling_rights else 0.0
    vec[offset + 3] = 1.0 if "k" in state.castling_rights else 0.0
    vec[offset + 4] = 1.0 if "q" in state.castling_rights else 0.0
    return vec


def simple_material_eval(state: GameState) -> float:
    """Material-only evaluation scaled to [-1, 1] for side to move."""
    score = 0
    for piece in state.board.squares:
        if piece == ".":
            continue
        value = PIECE_VALUES[piece.upper()]
        score += value if piece_color(piece) == "w" else -value
    # Normalize and orient to side to move
    max_score = 4000.0
    oriented = score / max_score
    return oriented if state.side_to_move == "w" else -oriented


def evaluate_position(state: GameState, model: Optional[Any] = None, device: str = "cpu") -> float:
    """Evaluate a position; falls back to material if no model given.

    `torch` is imported lazily only when a model is provided.
    """
    if model is None:
        return simple_material_eval(state)
    import torch

    dev = torch.device(device) if isinstance(device, str) else device
    model.to(dev)
    model.eval()
    with torch.no_grad():
        vec = encode_game_state(state, device=dev).unsqueeze(0)
        value = model(vec).item()
    return float(value)

