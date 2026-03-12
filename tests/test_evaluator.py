"""Tests for Evaluator functions."""

from gomoku.ai.evaluator import (
    COMBO_DOUBLE_OPEN_THREE,
    COMBO_OPEN_THREE_HALF_FOUR,
    SCORE_TABLE,
    evaluate,
    get_score,
)
from gomoku.board import Board
from gomoku.config import Player

# ---------------------------------------------------------------------------
# get_score
# ---------------------------------------------------------------------------


def test_five_in_a_row_score():
    max_score = SCORE_TABLE[(5, 0)]
    assert get_score(5, 0) == max_score
    assert get_score(5, 1) == max_score
    assert get_score(6, 0) == max_score


def test_open_four_score():
    assert get_score(4, 0) > get_score(4, 1) > 0


def test_open_three_score():
    assert get_score(3, 0) > get_score(3, 1) > 0


def test_double_blocked_score():
    assert get_score(3, 2) == 0
    assert get_score(2, 2) == 0


def test_score_ordering():
    assert get_score(5, 0) > get_score(4, 0) > get_score(4, 1)
    assert get_score(4, 1) >= get_score(3, 0)
    assert get_score(3, 0) > get_score(3, 1) >= get_score(2, 0) > 0


# ---------------------------------------------------------------------------
# evaluate — basic
# ---------------------------------------------------------------------------


def test_empty_board_score():
    board = Board()
    assert evaluate(board, Player.WHITE) == 0


def test_symmetric_evaluation():
    # 几何对称的位置（关于中心列对称），双方视角的净分应相等
    board = Board()
    board.place(7, 5, Player.BLACK)
    board.place(7, 9, Player.WHITE)
    score_as_black = evaluate(board, Player.BLACK)
    score_as_white = evaluate(board, Player.WHITE)
    assert score_as_black == score_as_white


def test_ai_advantage_with_more_pieces():
    board = Board()
    for col in range(3):
        board.place(7, col, Player.WHITE)
    board.place(0, 0, Player.BLACK)
    assert evaluate(board, Player.WHITE) > 0


def test_winning_position_high_score():
    board = Board()
    for col in range(5):
        board.place(7, col, Player.WHITE)
    assert evaluate(board, Player.WHITE) >= 100_000


def test_opponent_winning_gives_negative_score():
    board = Board()
    for col in range(5):
        board.place(7, col, Player.BLACK)
    assert evaluate(board, Player.WHITE) <= -100_000


# ---------------------------------------------------------------------------
# evaluate — defense weight
# ---------------------------------------------------------------------------


def test_defense_weight_amplifies_opponent_threat():
    # 对手有活四时，扣分应大于基础分（被 DEFENSE_WEIGHT 放大）
    board = Board()
    # 对手(BLACK)活四: (7,3)~(7,6)，两端 (7,2) 和 (7,7) 均空
    for col in range(3, 7):
        board.place(7, col, Player.BLACK)
    # AI 远处放一子，不影响对手棋型
    board.place(0, 0, Player.WHITE)

    score = evaluate(board, Player.WHITE)
    # 对手活四 10000 分被放大 1.5 倍 = -15000，AI 那一子远不能抵消
    assert score < -10_000


# ---------------------------------------------------------------------------
# evaluate — combo patterns
# ---------------------------------------------------------------------------


def test_double_open_three_bonus():
    # 构造双活三：两条活三交叉
    board = Board()
    # 横向活三: (7,6), (7,7), (7,8) — 两端 (7,5) 和 (7,9) 均空
    board.place(7, 6, Player.WHITE)
    board.place(7, 7, Player.WHITE)
    board.place(7, 8, Player.WHITE)
    # 纵向活三: (6,7), (7,7)[已有], (8,7) — 两端 (5,7) 和 (9,7) 均空
    board.place(6, 7, Player.WHITE)
    board.place(8, 7, Player.WHITE)

    score_with_combo = evaluate(board, Player.WHITE)

    # 对比：单条活三的分数（无组合加分）
    board2 = Board()
    board2.place(7, 6, Player.WHITE)
    board2.place(7, 7, Player.WHITE)
    board2.place(7, 8, Player.WHITE)
    score_single = evaluate(board2, Player.WHITE)

    # 双活三的分数应该远超单条活三（含 COMBO 加分）
    assert score_with_combo > score_single + COMBO_DOUBLE_OPEN_THREE // 2


def test_open_three_plus_half_four_bonus():
    # 构造活三+冲四组合
    board = Board()
    # 横向冲四: (7,0)~(7,3)，左端被边界封堵，右端 (7,4) 空 → 冲四
    for col in range(4):
        board.place(7, col, Player.WHITE)
    # 纵向活三: (6,2), (7,2)[已有], (8,2) — 两端 (5,2) 和 (9,2) 均空
    board.place(6, 2, Player.WHITE)
    board.place(8, 2, Player.WHITE)

    score = evaluate(board, Player.WHITE)

    # 分数应包含冲四(1000) + 活三(1000) + 组合加分(5000) 以及其他散棋
    assert score > SCORE_TABLE[(4, 1)] + SCORE_TABLE[(3, 0)] + COMBO_OPEN_THREE_HALF_FOUR // 2


def test_opponent_double_open_three_heavily_penalized():
    # 对手双活三时，AI 净分应严重为负（防守加权放大）
    board = Board()
    # 对手(BLACK)双活三
    board.place(7, 6, Player.BLACK)
    board.place(7, 7, Player.BLACK)
    board.place(7, 8, Player.BLACK)
    board.place(6, 7, Player.BLACK)
    board.place(8, 7, Player.BLACK)

    score = evaluate(board, Player.WHITE)
    # 对手双活三含组合加分，再被 DEFENSE_WEIGHT 放大，应严重为负
    assert score < -(2 * SCORE_TABLE[(3, 0)] + COMBO_DOUBLE_OPEN_THREE)
