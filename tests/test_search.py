import unittest

from chess_engine.game_state import GameState
from chess_engine.search import search_best_move


class SearchTests(unittest.TestCase):
    def test_search_returns_move(self):
        state = GameState.starting_state()
        move, score = search_best_move(state, model=None, depth=2)
        self.assertIsNotNone(move)
        self.assertTrue(isinstance(score, float))

    def test_search_depth_one(self):
        state = GameState.starting_state()
        move, _ = search_best_move(state, model=None, depth=1)
        self.assertIsNotNone(move)


if __name__ == "__main__":
    unittest.main()

