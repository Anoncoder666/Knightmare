import unittest

from chess_engine.board import START_FEN
from chess_engine.game_state import GameState


class BoardTests(unittest.TestCase):
    def test_starting_fen_roundtrip(self):
        start_fen_full = f"{START_FEN} w KQkq - 0 1"
        state = GameState.from_fen(start_fen_full)
        self.assertEqual(state.to_fen(), start_fen_full)

    def test_king_locations(self):
        state = GameState.starting_state()
        self.assertEqual(state.board.locate_king("w"), 60)
        self.assertEqual(state.board.locate_king("b"), 4)


if __name__ == "__main__":
    unittest.main()

