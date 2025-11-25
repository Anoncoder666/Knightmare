"""Game state container with move application and FEN handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .board import Board, START_FEN
from .utils import FILES, index_to_square, piece_color, square_to_index


@dataclass
class Move:
    from_square: int
    to_square: int
    promotion: Optional[str] = None
    is_castle: bool = False
    is_en_passant: bool = False
    captured: Optional[str] = None

    def __str__(self) -> str:
        promo = f"{self.promotion or ''}"
        return f"{index_to_square(self.from_square)}{index_to_square(self.to_square)}{promo}"


class GameState:
    """Holds current board state, history, and rules enforcement."""

    def __init__(
        self,
        board: Board,
        side_to_move: str = "w",
        castling_rights: str = "KQkq",
        en_passant: Optional[int] = None,
        halfmove_clock: int = 0,
        fullmove_number: int = 1,
    ) -> None:
        self.board = board
        self.side_to_move = side_to_move
        self.castling_rights = castling_rights if castling_rights != "-" else ""
        self.en_passant = en_passant
        self.halfmove_clock = halfmove_clock
        self.fullmove_number = fullmove_number
        self.history: List[Dict] = []
        self.repetition: Dict[str, int] = {}
        self.update_repetition()

    @classmethod
    def starting_state(cls) -> "GameState":
        fen = f"{START_FEN} w KQkq - 0 1"
        return cls.from_fen(fen)

    @classmethod
    def from_fen(cls, fen: str) -> "GameState":
        parts = fen.strip().split()
        if len(parts) != 6:
            raise ValueError("FEN must have 6 fields")
        board = Board.from_fen(parts[0])
        side = parts[1]
        castling = "" if parts[2] == "-" else parts[2]
        ep = None if parts[3] == "-" else square_to_index(parts[3])
        halfmove = int(parts[4])
        fullmove = int(parts[5])
        state = cls(board, side, castling, ep, halfmove, fullmove)
        return state

    def clone(self) -> "GameState":
        clone = GameState(
            self.board.copy(),
            self.side_to_move,
            self.castling_rights,
            self.en_passant,
            self.halfmove_clock,
            self.fullmove_number,
        )
        clone.repetition = self.repetition.copy()
        return clone

    def repetition_key(self) -> str:
        castling = self.castling_rights or "-"
        ep = "-" if self.en_passant is None else index_to_square(self.en_passant)
        return f"{self.board.to_fen()} {self.side_to_move} {castling} {ep}"

    def update_repetition(self) -> str:
        key = self.repetition_key()
        self.repetition[key] = self.repetition.get(key, 0) + 1
        return key

    def is_draw_by_repetition(self) -> bool:
        return self.repetition.get(self.repetition_key(), 0) >= 3

    def to_fen(self) -> str:
        castling = self.castling_rights or "-"
        ep = "-" if self.en_passant is None else index_to_square(self.en_passant)
        return f"{self.board.to_fen()} {self.side_to_move} {castling} {ep} {self.halfmove_clock} {self.fullmove_number}"

    def _side_king_square(self, color: str) -> int:
        return self.board.locate_king(color)

    def is_in_check(self, color: str) -> bool:
        king_sq = self._side_king_square(color)
        enemy = "b" if color == "w" else "w"
        return self.is_square_attacked(king_sq, enemy)

    def is_square_attacked(self, square: int, by_color: str) -> bool:
        """Check if a square is attacked by side."""
        row, col = divmod(square, 8)
        board = self.board.squares

        def on_board(r: int, c: int) -> bool:
            return 0 <= r < 8 and 0 <= c < 8

        # Pawn attacks
        pawn_dir = 1 if by_color == "w" else -1  # from target perspective
        for dc in (-1, 1):
            pr, pc = row + pawn_dir, col + dc
            if on_board(pr, pc):
                idx = pr * 8 + pc
                piece = board[idx]
                if by_color == "w" and piece == "P":
                    return True
                if by_color == "b" and piece == "p":
                    return True

        # Knights
        knight_steps = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for dr, dc in knight_steps:
            nr, nc = row + dr, col + dc
            if on_board(nr, nc):
                idx = nr * 8 + nc
                piece = board[idx]
                if by_color == "w" and piece == "N":
                    return True
                if by_color == "b" and piece == "n":
                    return True

        # Sliding pieces: bishops/queens
        diag_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in diag_dirs:
            nr, nc = row + dr, col + dc
            while on_board(nr, nc):
                idx = nr * 8 + nc
                piece = board[idx]
                if piece != ".":
                    if by_color == "w" and piece in ("B", "Q"):
                        return True
                    if by_color == "b" and piece in ("b", "q"):
                        return True
                    break
                nr += dr
                nc += dc

        # Sliding pieces: rooks/queens
        ortho_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in ortho_dirs:
            nr, nc = row + dr, col + dc
            while on_board(nr, nc):
                idx = nr * 8 + nc
                piece = board[idx]
                if piece != ".":
                    if by_color == "w" and piece in ("R", "Q"):
                        return True
                    if by_color == "b" and piece in ("r", "q"):
                        return True
                    break
                nr += dr
                nc += dc

        # King adjacency
        king_dirs = diag_dirs + ortho_dirs
        for dr, dc in king_dirs:
            nr, nc = row + dr, col + dc
            if on_board(nr, nc):
                idx = nr * 8 + nc
                piece = board[idx]
                if by_color == "w" and piece == "K":
                    return True
                if by_color == "b" and piece == "k":
                    return True
        return False

    def insufficient_material(self) -> bool:
        """Detect basic insufficient material (K vs K, K+minor vs K)."""
        pieces = [p for p in self.board.squares if p != "."]
        if all(p.upper() == "K" for p in pieces):
            return True
        minors = {"B", "N"}
        if len(pieces) == 3:
            # King and single minor vs king
            minor_pieces = [p for p in pieces if p.upper() in minors]
            if len(minor_pieces) == 1:
                return True
        return False

    def make_move(self, move: Move) -> None:
        board = self.board.squares
        moved_piece = board[move.from_square]
        if moved_piece == ".":
            raise ValueError("No piece on source square")
        target_piece = board[move.to_square]
        prev_castling = self.castling_rights
        prev_en_passant = self.en_passant
        prev_halfmove = self.halfmove_clock
        prev_fullmove = self.fullmove_number
        prev_side = self.side_to_move

        captured_piece: Optional[str] = target_piece if target_piece != "." else None
        ep_capture_square: Optional[int] = None
        rook_move: Optional[tuple[int, int, str]] = None

        # Handle en passant capture
        if move.is_en_passant:
            if self.side_to_move == "w":
                ep_capture_square = move.to_square + 8
            else:
                ep_capture_square = move.to_square - 8
            captured_piece = board[ep_capture_square]
            board[ep_capture_square] = "."

        # Move piece
        board[move.from_square] = "."
        placed_piece = moved_piece
        if move.promotion:
            placed_piece = move.promotion.upper() if self.side_to_move == "w" else move.promotion.lower()
        board[move.to_square] = placed_piece

        # Castling rook move
        if move.is_castle:
            if move.to_square == square_to_index("g1"):
                rook_from, rook_to = square_to_index("h1"), square_to_index("f1")
            elif move.to_square == square_to_index("c1"):
                rook_from, rook_to = square_to_index("a1"), square_to_index("d1")
            elif move.to_square == square_to_index("g8"):
                rook_from, rook_to = square_to_index("h8"), square_to_index("f8")
            elif move.to_square == square_to_index("c8"):
                rook_from, rook_to = square_to_index("a8"), square_to_index("d8")
            else:
                rook_from = rook_to = None  # type: ignore
            if rook_from is not None and rook_to is not None:
                rook_piece = board[rook_from]
                board[rook_from] = "."
                board[rook_to] = rook_piece
                rook_move = (rook_from, rook_to, rook_piece)

        # Update castling rights if king or rook moves/captured
        def remove_castle(right: str) -> None:
            self.castling_rights = self.castling_rights.replace(right, "")

        if moved_piece == "K":
            remove_castle("K")
            remove_castle("Q")
        elif moved_piece == "k":
            remove_castle("k")
            remove_castle("q")
        if moved_piece == "R":
            if move.from_square == square_to_index("a1"):
                remove_castle("Q")
            elif move.from_square == square_to_index("h1"):
                remove_castle("K")
        if moved_piece == "r":
            if move.from_square == square_to_index("a8"):
                remove_castle("q")
            elif move.from_square == square_to_index("h8"):
                remove_castle("k")
        if captured_piece == "R" and move.to_square == square_to_index("a1"):
            remove_castle("Q")
        if captured_piece == "R" and move.to_square == square_to_index("h1"):
            remove_castle("K")
        if captured_piece == "r" and move.to_square == square_to_index("a8"):
            remove_castle("q")
        if captured_piece == "r" and move.to_square == square_to_index("h8"):
            remove_castle("k")

        # En passant target
        self.en_passant = None
        if moved_piece.upper() == "P" and abs(move.to_square - move.from_square) == 16:
            self.en_passant = (move.to_square + move.from_square) // 2

        # Halfmove / fullmove
        if moved_piece.upper() == "P" or captured_piece:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        if self.side_to_move == "b":
            self.fullmove_number += 1

        # Switch side
        self.side_to_move = "b" if self.side_to_move == "w" else "w"

        rep_key = self.update_repetition()
        self.history.append(
            {
                "move": move,
                "captured": captured_piece,
                "moved_piece": moved_piece,
                "prev_castling": prev_castling,
                "prev_en_passant": prev_en_passant,
                "prev_halfmove": prev_halfmove,
                "prev_fullmove": prev_fullmove,
                "prev_side": prev_side,
                "rook_move": rook_move,
                "ep_capture_square": ep_capture_square,
                "rep_key": rep_key,
            }
        )

    def undo_move(self) -> None:
        if not self.history:
            return
        last = self.history.pop()
        move: Move = last["move"]
        board = self.board.squares

        # Remove repetition count for the position we are undoing
        rep_key = last["rep_key"]
        self.repetition[rep_key] = self.repetition.get(rep_key, 1) - 1
        if self.repetition[rep_key] <= 0:
            self.repetition.pop(rep_key, None)

        # Restore side before move
        self.side_to_move = last["prev_side"]
        self.castling_rights = last["prev_castling"]
        self.en_passant = last["prev_en_passant"]
        self.halfmove_clock = last["prev_halfmove"]
        self.fullmove_number = last["prev_fullmove"]

        # Undo board changes
        board[move.from_square] = last["moved_piece"]
        board[move.to_square] = last["captured"] or "."

        if move.is_en_passant and last["ep_capture_square"] is not None:
            board[move.to_square] = "."
            board[last["ep_capture_square"]] = last["captured"]

        if move.promotion:
            board[move.from_square] = last["moved_piece"]

        # Undo rook move on castling
        if move.is_castle and last["rook_move"]:
            rook_from, rook_to, rook_piece = last["rook_move"]
            board[rook_from] = rook_piece
            board[rook_to] = "."

    def legal_moves_available(self) -> bool:
        from .move_generation import generate_legal_moves

        return bool(generate_legal_moves(self))
