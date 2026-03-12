"""Tests for Evaluator functions."""

from gomoku.ai.evaluator import SCORE_TABLE, evaluate, get_score
from gomoku.board import Board
from gomoku.config import Player


# ---------------------------------------------------------------------------
# get_score
# ---------------------------------------------------------------------------

def test_five_in_a_row_score():
    # 五连无论封堵端数都是最高分
    max_score = SCORE_TABLE[(5, 0)]
    assert get_score(5, 0) == max_score
    assert get_score(5, 1) == max_score
    assert get_score(6, 0) == max_score  # 超过5也算赢


def test_open_four_score():
    # 活四(0封) >> 冲四(1封)
    assert get_score(4, 0) > get_score(4, 1) > 0


def test_open_three_score():
    # 活三(0封) >> 眠三(1封)
    assert get_score(3, 0) > get_score(3, 1) > 0


def test_double_blocked_score():
    # 两端都被堵，无价值
    assert get_score(3, 2) == 0
    assert get_score(2, 2) == 0


def test_score_ordering():
    # 五连 > 活四 > 冲四/活三 > 眠三 > 活二
    assert get_score(5, 0) > get_score(4, 0) > get_score(4, 1)
    assert get_score(4, 1) >= get_score(3, 0)
    assert get_score(3, 0) > get_score(3, 1) >= get_score(2, 0) > 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def test_empty_board_score():
    board = Board()
    assert evaluate(board, Player.WHITE) == 0


def test_symmetric_evaluation():
    # 双方各放同等棋型，净分应为0
    board = Board()
    board.place(7, 7, Player.BLACK)
    board.place(0, 0, Player.WHITE)
    score_as_black = evaluate(board, Player.BLACK)
    score_as_white = evaluate(board, Player.WHITE)
    assert score_as_black == -score_as_white


def test_ai_advantage_with_more_pieces():
    # AI(WHITE) 有三连，对手只有一子，AI 净分应为正
    board = Board()
    for col in range(3):
        board.place(7, col, Player.WHITE)
    board.place(0, 0, Player.BLACK)
    assert evaluate(board, Player.WHITE) > 0


def test_winning_position_high_score():
    # AI 五连时评分应达到最高档
    board = Board()
    for col in range(5):
        board.place(7, col, Player.WHITE)
    score = evaluate(board, Player.WHITE)
    assert score >= 100_000


def test_opponent_winning_gives_negative_score():
    # 对手五连时，对 AI 的净分应为负
    board = Board()
    for col in range(5):
        board.place(7, col, Player.BLACK)
    assert evaluate(board, Player.WHITE) <= -100_000
