"""Legal move generation."""

from __future__ import annotations

from typing import List

from .game_state import GameState, Move
from .utils import piece_color, square_to_index


def generate_pseudo_legal_moves(state: GameState) -> List[Move]:
    moves: List[Move] = []
    board = state.board.squares
    side = state.side_to_move

    def on_board(r: int, c: int) -> bool:
        return 0 <= r < 8 and 0 <= c < 8

    for idx, piece in enumerate(board):
        if piece == "." or piece_color(piece) != side:
            continue
        row, col = divmod(idx, 8)
        upper = piece.upper()
        enemy = "b" if side == "w" else "w"

        if upper == "P":
            dir = -1 if side == "w" else 1
            start_row = 6 if side == "w" else 1
            promo_row = 0 if side == "w" else 7

            # Single push
            fwd_r = row + dir
            if on_board(fwd_r, col):
                dest = fwd_r * 8 + col
                if board[dest] == ".":
                    if fwd_r == promo_row:
                        for promo in ("Q", "R", "B", "N"):
                            moves.append(Move(idx, dest, promotion=promo))
                    else:
                        moves.append(Move(idx, dest))

                    # Double push
                    if row == start_row:
                        dest2 = (row + 2 * dir) * 8 + col
                        if board[dest2] == ".":
                            moves.append(Move(idx, dest2))

            # Captures and en passant
            for dc in (-1, 1):
                cr, cc = row + dir, col + dc
                if not on_board(cr, cc):
                    continue
                target = cr * 8 + cc
                target_piece = board[target]
                if target_piece != "." and piece_color(target_piece) == enemy:
                    if cr == promo_row:
                        for promo in ("Q", "R", "B", "N"):
                            moves.append(Move(idx, target, promotion=promo, captured=target_piece))
                    else:
                        moves.append(Move(idx, target, captured=target_piece))
                # En passant
                if state.en_passant is not None and target == state.en_passant:
                    moves.append(Move(idx, target, is_en_passant=True))

        elif upper == "N":
            steps = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
            for dr, dc in steps:
                nr, nc = row + dr, col + dc
                if on_board(nr, nc):
                    dest = nr * 8 + nc
                    target_piece = board[dest]
                    if target_piece == "." or piece_color(target_piece) == enemy:
                        moves.append(Move(idx, dest, captured=target_piece if target_piece != "." else None))

        elif upper in ("B", "R", "Q"):
            directions = []
            if upper in ("B", "Q"):
                directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            if upper in ("R", "Q"):
                directions += [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                while on_board(nr, nc):
                    dest = nr * 8 + nc
                    target_piece = board[dest]
                    if target_piece == ".":
                        moves.append(Move(idx, dest))
                    else:
                        if piece_color(target_piece) == enemy:
                            moves.append(Move(idx, dest, captured=target_piece))
                        break
                    nr += dr
                    nc += dc

        elif upper == "K":
            king_dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for dr, dc in king_dirs:
                nr, nc = row + dr, col + dc
                if on_board(nr, nc):
                    dest = nr * 8 + nc
                    target_piece = board[dest]
                    if target_piece == "." or piece_color(target_piece) == enemy:
                        moves.append(Move(idx, dest, captured=target_piece if target_piece != "." else None))

            # Castling
            if side == "w" and row == 7 and col == 4:
                if "K" in state.castling_rights:
                    if board[square_to_index("f1")] == board[square_to_index("g1")] == ".":
                        if not state.is_square_attacked(idx, enemy) and not state.is_square_attacked(
                            square_to_index("f1"), enemy
                        ) and not state.is_square_attacked(square_to_index("g1"), enemy):
                            moves.append(Move(idx, square_to_index("g1"), is_castle=True))
                if "Q" in state.castling_rights:
                    if (
                        board[square_to_index("b1")] == board[square_to_index("c1")] == board[square_to_index("d1")] == "."
                    ):
                        if not state.is_square_attacked(idx, enemy) and not state.is_square_attacked(
                            square_to_index("d1"), enemy
                        ) and not state.is_square_attacked(square_to_index("c1"), enemy):
                            moves.append(Move(idx, square_to_index("c1"), is_castle=True))
            elif side == "b" and row == 0 and col == 4:
                if "k" in state.castling_rights:
                    if board[square_to_index("f8")] == board[square_to_index("g8")] == ".":
                        if not state.is_square_attacked(idx, enemy) and not state.is_square_attacked(
                            square_to_index("f8"), enemy
                        ) and not state.is_square_attacked(square_to_index("g8"), enemy):
                            moves.append(Move(idx, square_to_index("g8"), is_castle=True))
                if "q" in state.castling_rights:
                    if (
                        board[square_to_index("b8")] == board[square_to_index("c8")] == board[square_to_index("d8")] == "."
                    ):
                        if not state.is_square_attacked(idx, enemy) and not state.is_square_attacked(
                            square_to_index("d8"), enemy
                        ) and not state.is_square_attacked(square_to_index("c8"), enemy):
                            moves.append(Move(idx, square_to_index("c8"), is_castle=True))
    return moves


def generate_legal_moves(state: GameState) -> List[Move]:
    legal: List[Move] = []
    pseudo_moves = generate_pseudo_legal_moves(state)
    color = state.side_to_move
    for move in pseudo_moves:
        state.make_move(move)
        if not state.is_in_check(color):
            legal.append(move)
        state.undo_move()
    return legal

