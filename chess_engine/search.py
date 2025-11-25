"""Alpha-beta search leveraging neural evaluation."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from .evaluation import evaluate_position
from .game_state import GameState, Move
from .move_generation import generate_legal_moves
from .utils import PIECE_VALUES

MATE_VALUE = 10000.0


def order_moves(moves: list[Move]) -> list[Move]:
    """Simple move ordering: captures first by most valuable victim."""
    def move_score(move: Move) -> int:
        if move.captured:
            return PIECE_VALUES.get(move.captured.upper(), 0)
        return 0

    return sorted(moves, key=move_score, reverse=True)


def quiescence_search(
    state: GameState,
    alpha: float,
    beta: float,
    model,
    depth: int,
) -> float:
    stand_pat = evaluate_position(state, model)
    if stand_pat >= beta:
        return beta
    alpha = max(alpha, stand_pat)
    if depth <= 0:
        return stand_pat

    for move in order_moves(generate_legal_moves(state)):
        if not move.captured and not move.is_en_passant:
            continue
        state.make_move(move)
        score = -quiescence_search(state, -beta, -alpha, model, depth - 1)
        state.undo_move()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


def alpha_beta(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    model,
    quiescence_depth: int,
) -> float:
    if state.halfmove_clock >= 100 or state.insufficient_material() or state.is_draw_by_repetition():
        return 0.0

    if depth == 0:
        return quiescence_search(state, alpha, beta, model, quiescence_depth)

    legal_moves = order_moves(generate_legal_moves(state))
    if not legal_moves:
        if state.is_in_check(state.side_to_move):
            return -MATE_VALUE + (5 - depth)
        return 0.0

    value = -math.inf
    for move in legal_moves:
        state.make_move(move)
        score = -alpha_beta(state, depth - 1, -beta, -alpha, model, quiescence_depth)
        state.undo_move()
        value = max(value, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return value


def search_best_move(
    state: GameState,
    model=None,
    depth: int = 3,
    quiescence_depth: int = 3,
) -> Tuple[Optional[Move], float]:
    """Search best move using negamax alpha-beta."""
    best_move: Optional[Move] = None
    alpha = -math.inf
    beta = math.inf
    legal_moves = order_moves(generate_legal_moves(state))
    if not legal_moves:
        return None, 0.0

    for move in legal_moves:
        state.make_move(move)
        score = -alpha_beta(state, depth - 1, -beta, -alpha, model, quiescence_depth)
        state.undo_move()
        if score > alpha:
            alpha = score
            best_move = move
    return best_move, alpha

