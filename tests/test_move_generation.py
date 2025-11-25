import unittest

from chess_engine.game_state import GameState, Move
from chess_engine.move_generation import generate_legal_moves
from chess_engine.utils import square_to_index


class MoveGenerationTests(unittest.TestCase):
    def test_initial_position_moves(self):
        state = GameState.starting_state()
        moves = generate_legal_moves(state)
        self.assertEqual(len(moves), 20)

    def test_castling_moves_available(self):
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        state = GameState.from_fen(fen)
        moves = generate_legal_moves(state)
        castles = [m for m in moves if m.is_castle]
        self.assertTrue(any(m.to_square == square_to_index("g1") for m in castles))
        self.assertTrue(any(m.to_square == square_to_index("c1") for m in castles))

    def test_en_passant_generated(self):
        state = GameState.starting_state()
        # e2e4
        state.make_move(Move(square_to_index("e2"), square_to_index("e4")))
        # a7a6
        state.make_move(Move(square_to_index("a7"), square_to_index("a6")))
        # e4e5
        state.make_move(Move(square_to_index("e4"), square_to_index("e5")))
        # d7d5 creating en passant target
        state.make_move(Move(square_to_index("d7"), square_to_index("d5")))
        moves = generate_legal_moves(state)
        ep_moves = [m for m in moves if m.is_en_passant]
        self.assertTrue(any(m.from_square == square_to_index("e5") for m in ep_moves))


if __name__ == "__main__":
    unittest.main()

