import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stratnet4 import (Board, MinimaxAgent, AlphaBetaAgent, MCTSAgent,
                       RandomAgent, random_position, play_game)
from stratnet4.agents import WINDOWS, WIN


class BoardTests(unittest.TestCase):
    def test_window_count(self):
        self.assertEqual(len(WINDOWS), 69)

    def test_vertical_win(self):
        b = Board.from_moves([0, 1, 0, 1, 0, 1])
        self.assertTrue(b.is_winning_move(0))
        b.play(0)
        self.assertTrue(b.last_mover_won())

    def test_horizontal_win(self):
        b = Board.from_moves([0, 0, 1, 1, 2, 2])
        self.assertTrue(b.is_winning_move(3))

    def test_diagonal_win(self):
        # X builds a diagonal 0,1,2,3
        b = Board.from_moves([0, 1, 1, 2, 2, 3, 2, 3, 3, 6])
        self.assertTrue(b.is_winning_move(3))

    def test_full_column(self):
        b = Board.from_moves([0] * 6)
        self.assertFalse(b.can_play(0))
        self.assertNotIn(0, b.legal_moves())

    def test_no_wrap_between_columns(self):
        # stones at top of col 0 and bottom of col 1 must not connect
        b = Board.from_moves([0, 1, 0, 1, 0, 2, 1, 0, 6, 0])
        self.assertFalse(b.last_mover_won())


class AgentTests(unittest.TestCase):
    def test_alpha_beta_matches_minimax_value(self):
        rng = random.Random(1)
        mm, ab = MinimaxAgent(depth=4, seed=0), AlphaBetaAgent(depth=4, seed=0)
        for _ in range(15):
            seq = random_position(rng, rng.randint(3, 12))
            if seq is None:
                continue
            b = Board.from_moves(seq)
            mm.choose(b)
            ab.choose(b)
            self.assertEqual(mm.last_stats["score"], ab.last_stats["score"])
            self.assertLessEqual(ab.last_stats["nodes"], mm.last_stats["nodes"])

    def test_takes_immediate_win(self):
        b = Board.from_moves([0, 6, 1, 6, 2, 5])        # X wins at col 3
        for ag in (MinimaxAgent(2), AlphaBetaAgent(2), MCTSAgent(iterations=800, seed=1)):
            self.assertEqual(ag.choose(b), 3, ag.name)

    def test_blocks_immediate_loss(self):
        b = Board.from_moves([0, 6, 1, 6, 2])           # O to move, X threatens col 3
        for ag in (MinimaxAgent(2), AlphaBetaAgent(2), MCTSAgent(iterations=1500, seed=1)):
            self.assertEqual(ag.choose(b), 3, ag.name)

    def test_time_limited_search_returns_legal_move(self):
        ag = AlphaBetaAgent(depth=None, time_limit=0.02, seed=0)
        self.assertTrue(Board().can_play(ag.choose(Board())))
        self.assertGreaterEqual(ag.last_stats["depth"], 1)

    def test_alphabeta_crushes_random(self):
        res = [play_game(AlphaBetaAgent(3, seed=i), RandomAgent(seed=i), ())[0] for i in range(6)]
        self.assertEqual(res, [0] * 6)


if __name__ == "__main__":
    unittest.main()
