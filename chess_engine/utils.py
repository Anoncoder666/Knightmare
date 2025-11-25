"""Utility helpers for board coordinates and pieces."""

from __future__ import annotations

from typing import Optional

FILES = "abcdefgh"
RANKS = "12345678"


def index_to_square(index: int) -> str:
    """Convert 0-63 board index to algebraic square like 'e4'."""
    row, col = divmod(index, 8)
    return f"{FILES[col]}{8 - row}"


def square_to_index(square: str) -> int:
    """Convert algebraic square (e.g., 'e4') to 0-63 index."""
    if len(square) != 2 or square[0] not in FILES or square[1] not in RANKS:
        raise ValueError(f"Invalid square: {square}")
    file = FILES.index(square[0])
    rank = int(square[1])
    row = 8 - rank
    return row * 8 + file


def piece_color(piece: str) -> Optional[str]:
    """Return 'w' for white piece, 'b' for black piece, or None for empty."""
    if piece == ".":
        return None
    return "w" if piece.isupper() else "b"


def is_white(piece: str) -> bool:
    return piece_color(piece) == "w"


def is_black(piece: str) -> bool:
    return piece_color(piece) == "b"


PIECE_VALUES = {
    "P": 100,
    "N": 320,
    "B": 330,
    "R": 500,
    "Q": 900,
    "K": 0,
}

