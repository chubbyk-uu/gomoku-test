"""Tests for pattern-based Evaluator."""

import random

from gomoku.ai.evaluator import (
    Shape,
    _count_shapes,
    _count_shapes_after_move,
    _count_shapes_legacy,
    _extract_line,
    _match_shapes,
    evaluate,
)
from gomoku.board import Board
from gomoku.config import Player

# ===================================================================
# 辅助函数
# ===================================================================


def _board_with_pieces(pieces: list[tuple[int, int, Player]]) -> Board:
    """快速创建带棋子的棋盘。"""
    board = Board()
    for r, c, p in pieces:
        board.place(r, c, p)
    return board


# ===================================================================
# _extract_line 基础测试
# ===================================================================


def test_extract_line_center_of_empty_board():
    board = Board()
    grid = board.grid.tolist()
    # 中心空位，应全为 0
    line = _extract_line(grid, 7, 7, 0, 1, int(Player.BLACK))
    assert len(line) == 9
    assert all(v == 0 for v in line)


def test_extract_line_boundary_fills_opponent():
    board = Board()
    board.place(0, 0, Player.BLACK)
    grid = board.grid.tolist()
    # 向右提取，左侧越界应为 2（对方）
    line = _extract_line(grid, 0, 0, 0, 1, int(Player.BLACK))
    assert len(line) == 9
    # index 0~3 是越界 → 对方
    for i in range(4):
        assert line[i] == 2
    assert line[4] == 1  # 中心是己方


# ===================================================================
# _match_shapes — FIVE
# ===================================================================


def test_match_five_in_center():
    # _____XXXXX → 在 line[4] 为中心
    line = [0, 0, 0, 0, 1, 1, 1, 1, 1]  # index 4-8 全 X
    # 不对: index 0-3 空，4-8 X。line[4]=X -> 连五
    # 实际上需要 5 个连续 X 包含 index 4
    assert Shape.FIVE in _match_shapes(line)


def test_match_five_horizontal():
    board = _board_with_pieces([(7, c, Player.BLACK) for c in range(5)])
    counts = _count_shapes(board, Player.BLACK)
    assert counts[Shape.FIVE] >= 1


# ===================================================================
# _match_shapes — OPEN_FOUR
# ===================================================================


def test_match_open_four():
    # _XXXX_ : line index 中有 E X X X X E
    line = [2, 0, 0, 1, 1, 1, 1, 0, 2]
    shapes = _match_shapes(line)
    assert Shape.OPEN_FOUR in shapes


def test_open_four_on_board():
    # 活四: 4连子两端都空
    board = _board_with_pieces([(7, c, Player.WHITE) for c in range(3, 7)])
    # (7,2) 和 (7,7) 都空 → 活四
    counts = _count_shapes(board, Player.WHITE)
    assert counts[Shape.OPEN_FOUR] >= 1


# ===================================================================
# _match_shapes — HALF_FOUR（连冲四和跳冲四）
# ===================================================================


def test_match_half_four_blocked_left():
    # OXXXX_ : O在左
    line = [0, 0, 0, 2, 1, 1, 1, 1, 0]
    shapes = _match_shapes(line)
    assert Shape.HALF_FOUR in shapes


def test_match_half_four_blocked_right():
    # _XXXXO : O在右
    line = [0, 0, 1, 1, 1, 1, 2, 0, 0]
    shapes = _match_shapes(line)
    assert Shape.HALF_FOUR in shapes


def test_match_half_four_jump_x_xxx():
    # X_XXX: 跳冲四
    line = [0, 0, 0, 0, 1, 0, 1, 1, 1]
    # index 4=X, 5=E, 6=X, 7=X, 8=X → X_XXX
    shapes = _match_shapes(line)
    assert Shape.HALF_FOUR in shapes


def test_match_half_four_jump_xxx_x():
    # XXX_X: 跳冲四
    # 直接构造棋盘局面比 line 中心点测试更可靠
    board = Board()
    # XXX_X 横向: (7,3)(7,4)(7,5) 空 (7,7)
    board.place(7, 3, Player.BLACK)
    board.place(7, 4, Player.BLACK)
    board.place(7, 5, Player.BLACK)
    board.place(7, 7, Player.BLACK)
    counts = _count_shapes(board, Player.BLACK)
    assert counts[Shape.HALF_FOUR] >= 1


def test_match_half_four_jump_xx_xx():
    # XX_XX: 跳冲四
    board = Board()
    board.place(7, 3, Player.BLACK)
    board.place(7, 4, Player.BLACK)
    # 空 (7,5)
    board.place(7, 6, Player.BLACK)
    board.place(7, 7, Player.BLACK)
    counts = _count_shapes(board, Player.BLACK)
    assert counts[Shape.HALF_FOUR] >= 1


def test_half_four_edge():
    # 边界连冲四：XXXX_ 在棋盘左边界
    board = Board()
    for c in range(4):
        board.place(7, c, Player.BLACK)
    # 左端被边界封堵 → 冲四
    counts = _count_shapes(board, Player.BLACK)
    assert counts[Shape.HALF_FOUR] >= 1


# ===================================================================
# _match_shapes — OPEN_THREE（连活三和跳活三）
# ===================================================================


def test_match_open_three():
    # _XXX_ 两端都空，且外侧也空 → 活三
    board = _board_with_pieces([(7, c, Player.WHITE) for c in range(6, 9)])
    # (7,5) 和 (7,9) 空，且 (7,4) 和 (7,10) 也空 → __XXX_ 或 _XXX__
    counts = _count_shapes(board, Player.WHITE)
    assert counts[Shape.OPEN_THREE] >= 1


def test_match_open_three_jump_x_xx():
    # _X_XX_: 跳活三
    board = Board()
    board.place(7, 4, Player.BLACK)
    # 空 (7,5)
    board.place(7, 6, Player.BLACK)
    board.place(7, 7, Player.BLACK)
    # (7,3) 和 (7,8) 空 → _X_XX_
    counts = _count_shapes(board, Player.BLACK)
    assert counts[Shape.OPEN_THREE] >= 1


def test_match_open_three_jump_xx_x():
    # _XX_X_: 跳活三
    board = Board()
    board.place(7, 4, Player.BLACK)
    board.place(7, 5, Player.BLACK)
    # 空 (7,6)
    board.place(7, 7, Player.BLACK)
    # (7,3) 和 (7,8) 空 → _XX_X_
    counts = _count_shapes(board, Player.BLACK)
    assert counts[Shape.OPEN_THREE] >= 1


# ===================================================================
# _match_shapes — HALF_THREE（连眠三和跳眠三）
# ===================================================================


def test_match_half_three_blocked():
    # OXXX__ : 一端被对方封堵
    board = Board()
    board.place(7, 3, Player.WHITE)  # 对方封堵
    board.place(7, 4, Player.BLACK)
    board.place(7, 5, Player.BLACK)
    board.place(7, 6, Player.BLACK)
    # (7,7) 和 (7,8) 空
    counts = _count_shapes(board, Player.BLACK)
    assert counts[Shape.HALF_THREE] >= 1


def test_match_half_three_jump():
    # OX_XX_: 跳眠三
    board = Board()
    board.place(7, 3, Player.WHITE)  # 对方封堵
    board.place(7, 4, Player.BLACK)
    # 空 (7,5)
    board.place(7, 6, Player.BLACK)
    board.place(7, 7, Player.BLACK)
    # (7,8) 空
    counts = _count_shapes(board, Player.BLACK)
    assert counts[Shape.HALF_THREE] >= 1


# ===================================================================
# _match_shapes — OPEN_TWO 和 HALF_TWO
# ===================================================================


def test_match_open_two():
    # __XX__: 活二
    board = _board_with_pieces([(7, 6, Player.BLACK), (7, 7, Player.BLACK)])
    counts = _count_shapes(board, Player.BLACK)
    assert counts[Shape.OPEN_TWO] >= 1


def test_match_open_two_jump():
    # _X_X_: 跳活二
    board = Board()
    board.place(7, 5, Player.BLACK)
    board.place(7, 7, Player.BLACK)
    # (7,4), (7,6), (7,8) 空 → _X_X_
    counts = _count_shapes(board, Player.BLACK)
    assert counts[Shape.OPEN_TWO] >= 1


def test_match_half_two():
    # OXX___: 眠二（一端被对方封堵）
    board = Board()
    board.place(7, 3, Player.WHITE)  # 封堵
    board.place(7, 4, Player.BLACK)
    board.place(7, 5, Player.BLACK)
    counts = _count_shapes(board, Player.BLACK)
    assert counts[Shape.HALF_TWO] >= 1


def test_match_half_two_jump():
    # OX_X__: 跳眠二
    board = Board()
    board.place(7, 3, Player.WHITE)  # 封堵
    board.place(7, 4, Player.BLACK)
    # 空 (7,5)
    board.place(7, 6, Player.BLACK)
    counts = _count_shapes(board, Player.BLACK)
    assert counts[Shape.HALF_TWO] >= 1


def test_count_shapes_after_move_matches_place_and_recount():
    board = _board_with_pieces(
        [
            (7, 7, Player.BLACK),
            (7, 8, Player.BLACK),
            (8, 7, Player.BLACK),
            (6, 6, Player.WHITE),
            (6, 8, Player.WHITE),
            (8, 8, Player.WHITE),
        ]
    )

    hypothetical = _count_shapes_after_move(board, Player.BLACK, 7, 6)

    board.place(7, 6, Player.BLACK)
    try:
        actual = _count_shapes(board, Player.BLACK)
    finally:
        board.undo()

    assert hypothetical == actual


def test_incremental_count_shapes_matches_legacy_on_fixed_position():
    board = Board()
    pieces = [
        (7, 7, Player.BLACK),
        (7, 8, Player.WHITE),
        (8, 7, Player.BLACK),
        (8, 8, Player.WHITE),
        (6, 7, Player.BLACK),
        (9, 8, Player.WHITE),
        (7, 6, Player.BLACK),
        (8, 9, Player.WHITE),
    ]
    for r, c, p in pieces:
        board.place(r, c, p)

    for player in (Player.BLACK, Player.WHITE):
        assert _count_shapes(board, player) == _count_shapes_legacy(board, player)


def test_incremental_count_shapes_matches_legacy_on_random_positions():
    random.seed(0)
    cells = [(r, c) for r in range(15) for c in range(15)]

    for stones in range(2, 12, 2):
        for _ in range(20):
            board = Board()
            player = Player.BLACK
            for row, col in random.sample(cells, stones):
                board.place(row, col, player)
                player = Player.WHITE if player == Player.BLACK else Player.BLACK

            for side in (Player.BLACK, Player.WHITE):
                assert _count_shapes(board, side) == _count_shapes_legacy(board, side)


# ===================================================================
# evaluate — 基础
# ===================================================================


def test_empty_board_score():
    board = Board()
    assert evaluate(board, Player.WHITE) == 0


def test_symmetric_evaluation():
    """黑白互换后评分符号相反。"""
    board = Board()
    board.place(7, 5, Player.BLACK)
    board.place(7, 9, Player.WHITE)
    score_as_black = evaluate(board, Player.BLACK)
    score_as_white = evaluate(board, Player.WHITE)
    # 对称位置，双方视角评分应相等
    assert score_as_black == score_as_white


def test_sign_inversion():
    """同一棋局，双方视角评分符号相反。"""
    board = Board()
    for c in range(6, 9):
        board.place(7, c, Player.BLACK)
    score_b = evaluate(board, Player.BLACK)
    score_w = evaluate(board, Player.WHITE)
    # 黑方视角 > 0，白方视角 < 0
    assert score_b > 0
    assert score_w < 0


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


def test_ai_advantage_with_more_pieces():
    board = Board()
    for col in range(3):
        board.place(7, col, Player.WHITE)
    board.place(0, 0, Player.BLACK)
    assert evaluate(board, Player.WHITE) > 0


# ===================================================================
# evaluate — 防守加权
# ===================================================================


def test_defense_weight_amplifies_opponent_threat():
    board = Board()
    # 对手活四
    for col in range(3, 7):
        board.place(7, col, Player.BLACK)
    board.place(0, 0, Player.WHITE)

    score = evaluate(board, Player.WHITE)
    # 对手活四 50000 × 1.2 = 60000 远超 AI 单子
    assert score < -10_000


# ===================================================================
# evaluate — 组合棋型
# ===================================================================


def test_double_open_three_combo():
    """双活三组合 → 10000 分。"""
    board = Board()
    # 横向活三: (7,6), (7,7), (7,8)
    board.place(7, 6, Player.WHITE)
    board.place(7, 7, Player.WHITE)
    board.place(7, 8, Player.WHITE)
    # 纵向活三: (6,7), (7,7)[已有], (8,7)
    board.place(6, 7, Player.WHITE)
    board.place(8, 7, Player.WHITE)

    score = evaluate(board, Player.WHITE)
    # 双活三 = 10000
    assert score >= 10_000


def test_half_four_plus_open_three_combo():
    """冲四 + 活三组合 → 10000 分。"""
    board = Board()
    # 横向冲四: (7,0)~(7,3)，左端被边界封堵
    for col in range(4):
        board.place(7, col, Player.WHITE)
    # 纵向活三: (6,2), (7,2)[已有], (8,2)
    board.place(6, 2, Player.WHITE)
    board.place(8, 2, Player.WHITE)

    score = evaluate(board, Player.WHITE)
    assert score >= 5_000


def test_opponent_double_open_three_penalized():
    """对手双活三应严重扣分。"""
    board = Board()
    board.place(7, 6, Player.BLACK)
    board.place(7, 7, Player.BLACK)
    board.place(7, 8, Player.BLACK)
    board.place(6, 7, Player.BLACK)
    board.place(8, 7, Player.BLACK)

    score = evaluate(board, Player.WHITE)
    # 对手双活三 10000 × 1.2 = 12000
    assert score < -5_000


def test_open_four_is_winning():
    """活四 → 50000 分（必胜）。"""
    board = Board()
    for col in range(4, 8):
        board.place(7, col, Player.WHITE)
    counts = _count_shapes(board, Player.WHITE)
    assert counts[Shape.OPEN_FOUR] >= 1
    score = evaluate(board, Player.WHITE)
    assert score >= 50_000


def test_four_three_scores_higher_than_lone_half_four():
    """冲四+活三应明显优于孤立冲四。"""
    half_four_board = Board()
    for col in range(4):
        half_four_board.place(7, col, Player.WHITE)

    combo_board = Board()
    for col in range(4):
        combo_board.place(7, col, Player.WHITE)
    combo_board.place(6, 2, Player.WHITE)
    combo_board.place(8, 2, Player.WHITE)

    assert evaluate(combo_board, Player.WHITE) > evaluate(half_four_board, Player.WHITE)


def test_lone_half_four_is_kept_below_combo_threshold():
    """孤立冲四不应被评成接近强制杀组合。"""
    board = Board()
    for col in range(4):
        board.place(7, col, Player.WHITE)

    score = evaluate(board, Player.WHITE)
    assert score < 5_000
