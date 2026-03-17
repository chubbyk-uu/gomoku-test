"""Tests for standalone VCF profiling and benchmark helpers."""

from gomoku.ai.vcf import VCFSolver
from gomoku.board import Board
from gomoku.config import Player


def test_vcf_stats_populated_after_winning_search():
    board = Board()
    for row, col in [(7, 0), (7, 1), (7, 2), (5, 3), (6, 3), (8, 3)]:
        board.place(row, col, Player.WHITE)
    board.place(14, 14, Player.BLACK)

    solver = VCFSolver()
    move = solver.find_winning_move(board, Player.WHITE, max_depth=8)

    assert move == (7, 3)
    assert solver.last_stats.mode == "win"
    assert solver.last_stats.elapsed_s >= 0.0
    assert solver.last_stats.nodes > 0
    assert solver.last_stats.attack_candidates > 0
    assert solver.last_stats.classified_moves > 0
    assert solver.last_stats.max_depth_reached > 0


def test_vcf_stats_populated_after_blocking_search():
    board = Board()
    for row, col in [(7, 0), (7, 1), (7, 2), (5, 3), (6, 3), (8, 3)]:
        board.place(row, col, Player.BLACK)
    board.place(14, 14, Player.WHITE)

    solver = VCFSolver()
    move = solver.find_blocking_move(board, Player.WHITE, max_depth=8)

    assert move == (7, 3)
    assert solver.last_stats.mode == "block"
    assert solver.last_stats.elapsed_s >= 0.0
    assert solver.last_stats.nodes > 0
    assert solver.last_stats.defense_candidates > 0
