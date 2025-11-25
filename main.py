"""Command-line interface to play against the chess engine."""

from __future__ import annotations

import argparse
from typing import Optional, Any

from chess_engine.evaluation import evaluate_position
from chess_engine.game_state import GameState, Move
from chess_engine.move_generation import generate_legal_moves
from chess_engine.search import search_best_move
from chess_engine.utils import index_to_square, square_to_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play chess against a PyTorch-powered engine.")
    parser.add_argument("--play", action="store_true", help="Play a game against the engine.")
    parser.add_argument("--fen", type=str, default=None, help="Start from a custom FEN.")
    parser.add_argument("--depth", type=int, default=3, help="Search depth for the engine.")
    parser.add_argument("--engine-color", choices=["white", "black"], default="black", help="Engine side.")
    parser.add_argument("--load-model", type=str, default=None, help="Path to a trained model checkpoint.")
    return parser.parse_args()


def load_eval_model(path: Optional[str]) -> Optional[Any]:
    """Lazily load the evaluation model only when requested.

    This avoids importing `torch` at module import time so the CLI can run
    without PyTorch installed when no model is provided.
    """
    if path:
        print(f"Loading model from {path} ...")
        # Import lazily to avoid hard dependency on torch at startup
        from chess_engine.model import load_model

        return load_model(path, device="cpu")
    return None


def find_move_from_input(move_input: str, legal_moves: list[Move]) -> Optional[Move]:
    move_input = move_input.strip().lower()
    if len(move_input) not in (4, 5):
        return None
    try:
        from_sq = square_to_index(move_input[:2])
        to_sq = square_to_index(move_input[2:4])
    except ValueError:
        return None
    promo = move_input[4] if len(move_input) == 5 else None
    for move in legal_moves:
        if move.from_square == from_sq and move.to_square == to_sq:
            if promo is None or (move.promotion and move.promotion.lower() == promo.lower()):
                return move
    return None


def print_board(state: GameState) -> None:
    print(state.board.pretty())
    print(f"Side to move: {'White' if state.side_to_move == 'w' else 'Black'}")


def play(args: argparse.Namespace) -> None:
    state = GameState.from_fen(args.fen) if args.fen else GameState.starting_state()
    model = load_eval_model(args.load_model)

    engine_side = "w" if args.engine_color == "white" else "b"

    while True:
        print_board(state)
        legal_moves = generate_legal_moves(state)
        if not legal_moves:
            if state.is_in_check(state.side_to_move):
                winner = "Black" if state.side_to_move == "w" else "White"
                print(f"Checkmate! {winner} wins.")
            else:
                print("Stalemate.")
            break
        if state.halfmove_clock >= 100 or state.insufficient_material() or state.is_draw_by_repetition():
            print("Draw by rule.")
            break

        if state.side_to_move == engine_side:
            best_move, score = search_best_move(state, model=model, depth=args.depth)
            if best_move is None:
                print("Engine resigns.")
                break
            state.make_move(best_move)
            print(f"Engine plays: {best_move} (eval {score:.3f})")
        else:
            move_input = input("Your move (e.g., e2e4, 'quit' to exit): ").strip()
            if move_input.lower() in {"quit", "exit", "resign"}:
                print("Game ended by user.")
                break
            chosen_move = find_move_from_input(move_input, legal_moves)
            if chosen_move is None:
                print("Illegal or unrecognized move, try again.")
                continue
            state.make_move(chosen_move)


def main() -> None:
    args = parse_args()
    if args.play:
        play(args)
    else:
        print("No action specified. Use --play to start a game against the engine.")


if __name__ == "__main__":
    main()
