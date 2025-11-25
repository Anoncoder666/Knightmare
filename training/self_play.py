"""Simple self-play data generator."""

from __future__ import annotations

import argparse
import json
import os
from typing import List

from chess_engine.game_state import GameState
from chess_engine.search import search_best_move


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate self-play data.")
    parser.add_argument("--games", type=int, default=5, help="Number of games to play.")
    parser.add_argument("--max-moves", type=int, default=60, help="Maximum moves per game.")
    parser.add_argument("--depth", type=int, default=2, help="Search depth for self-play.")
    parser.add_argument("--output", type=str, default="data/selfplay.jsonl", help="Output JSONL file.")
    return parser.parse_args()


def game_result_value(state: GameState, legal_moves: List) -> int:
    if legal_moves:
        return 0
    if state.is_in_check(state.side_to_move):
        return -1 if state.side_to_move == "w" else 1
    return 0


def play_self_game(depth: int, max_moves: int):
    state = GameState.starting_state()
    history: List[str] = []
    for _ in range(max_moves):
        legal_moves = []
        try:
            from chess_engine.move_generation import generate_legal_moves

            legal_moves = generate_legal_moves(state)
        except Exception:
            break
        if not legal_moves or state.halfmove_clock >= 100 or state.insufficient_material():
            break
        move, _ = search_best_move(state, model=None, depth=depth)
        if move is None:
            break
        history.append(state.to_fen())
        state.make_move(move)
    # Final result for labeling
    final_moves = []
    from chess_engine.move_generation import generate_legal_moves

    final_moves = generate_legal_moves(state)
    result = game_result_value(state, final_moves)
    return history, result


def save_history(history: List[str], result: int, output: str) -> None:
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "a", encoding="utf-8") as f:
        for fen in history:
            label = result if fen.split()[1] == "w" else -result
            f.write(json.dumps({"fen": fen, "value": label}) + "\n")


def main() -> None:
    args = parse_args()
    for _ in range(args.games):
        history, result = play_self_game(args.depth, args.max_moves)
        save_history(history, result, args.output)
    print(f"Wrote self-play games to {args.output}")


if __name__ == "__main__":
    main()

