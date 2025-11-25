"""Board representation and FEN helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .utils import FILES, index_to_square


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


@dataclass
class Board:
    """Simple 8x8 board represented as a flat list of 64 characters."""

    squares: List[str]

    @classmethod
    def starting_board(cls) -> "Board":
        return cls.from_fen(START_FEN)

    @classmethod
    def from_fen(cls, board_fen: str) -> "Board":
        rows = board_fen.split("/")
        if len(rows) != 8:
            raise ValueError("Invalid FEN board section")
        squares: List[str] = []
        for row in rows:
            for ch in row:
                if ch.isdigit():
                    squares.extend(["."] * int(ch))
                else:
                    squares.append(ch)
        if len(squares) != 64:
            raise ValueError("Invalid FEN board section length")
        return cls(squares)

    def copy(self) -> "Board":
        return Board(self.squares.copy())

    def to_fen(self) -> str:
        fen_rows = []
        for r in range(8):
            row = self.squares[r * 8 : (r + 1) * 8]
            empty = 0
            fen_row = ""
            for piece in row:
                if piece == ".":
                    empty += 1
                else:
                    if empty:
                        fen_row += str(empty)
                        empty = 0
                    fen_row += piece
            if empty:
                fen_row += str(empty)
            fen_rows.append(fen_row)
        return "/".join(fen_rows)

    def __getitem__(self, index: int) -> str:
        return self.squares[index]

    def __setitem__(self, index: int, value: str) -> None:
        self.squares[index] = value

    def locate_king(self, color: str) -> int:
        target = "K" if color == "w" else "k"
        for idx, piece in enumerate(self.squares):
            if piece == target:
                return idx
        raise ValueError(f"No king found for {color}")

    def __str__(self) -> str:
        return self.pretty()

    def pretty(self) -> str:
        """Return ASCII board."""
        # ANSI background colors (256-color). Two shades of brown for the checkerboard.
        BG_LIGHT = "\x1b[48;5;180m"
        BG_DARK = "\x1b[48;5;94m"
        BG_BORDER = "\x1b[48;5;16m"  # black border
        RESET = "\x1b[0m"

        # Visible widths: left border (2) + rank label area (2) + 8 squares * 2 chars + right border (2)
        visible_width = 2 + 2 + 8 * 2 + 2

        # Unicode piece symbols (white: uppercase, black: lowercase)
        PIECE_UNICODE = {
            "K": "\u2654",
            "Q": "\u2655",
            "R": "\u2656",
            "B": "\u2657",
            "N": "\u2658",
            "P": "\u2659",
            "k": "\u265A",
            "q": "\u265B",
            "r": "\u265C",
            "b": "\u265D",
            "n": "\u265E",
            "p": "\u265F",
        }

        lines: List[str] = []
        # Top border (one row)
        top_border = BG_BORDER + (" " * visible_width) + RESET
        lines.append(top_border)

        for r in range(8):
            row = self.squares[r * 8 : (r + 1) * 8]
            rank = 8 - r
            cells: List[str] = []
            for c, piece in enumerate(row):
                # choose background based on checker pattern
                bg = BG_LIGHT if (r + c) % 2 == 0 else BG_DARK
                # show unicode piece or space; apply background to the leading space and the piece
                ch = PIECE_UNICODE.get(piece, " ") if piece != "." else " "
                cell = f"{bg} {ch}{RESET}"
                cells.append(cell)
            # Left border (2 black spaces), then rank label (with border background), then board cells, then right border
            left_border = BG_BORDER + "  "
            right_border = BG_BORDER + "  " + RESET
            # rank label should be inside the border background; remove extra space after rank
            lines.append(left_border + f"{rank}" + "".join(cells) + right_border)

        # Footer: two rows for bottom border area before file labels
        # First print the file label row (inside the border)
        # File row: include labels inside the border background (aligned under squares)
        file_content = " " + " ".join(FILES) + " "
        file_row = BG_BORDER + "  " + file_content + "  " + RESET
        lines.append(file_row)

        # Bottom border: full background row (no half-block glyphs)
        bottom_border = BG_BORDER + (" " * visible_width) + RESET
        lines.append(bottom_border)

        return "\n".join(lines)

