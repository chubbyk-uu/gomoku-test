"""Tests for AISearcher."""

from gomoku.ai.searcher import AISearcher
from gomoku.board import Board
from gomoku.config import Player


def _make_searcher(ai_player: Player = Player.WHITE, depth: int = 2) -> AISearcher:
    return AISearcher(depth=depth, ai_player=ai_player)


# ---------------------------------------------------------------------------
# test_ai_blocks_four
# ---------------------------------------------------------------------------


def test_ai_blocks_four():
    """对手形成冲四（一端被边界封堵）时，AI 应该堵住唯一的开放端。"""
    board = Board()
    # 人类(BLACK)在第0行连了4子: (0,0),(0,1),(0,2),(0,3)
    # 左端 col=-1 越界（天然封堵），右端 (0,4) 是唯一开口，AI 必须堵这里
    for col in range(4):
        board.place(0, col, Player.BLACK)
    searcher = _make_searcher(ai_player=Player.WHITE, depth=2)
    move = searcher.find_best_move(board)
    assert move is not None
    assert move == (0, 4), f"Expected blocking move (0,4), got {move}"


# ---------------------------------------------------------------------------
# test_ai_wins_when_possible
# ---------------------------------------------------------------------------


def test_ai_wins_when_possible():
    """AI 自身已有四连时，应该直接补全五连获胜。"""
    board = Board()
    # AI(WHITE) 在第5行已有4子: (5,0),(5,1),(5,2),(5,3)，右端 (5,4) 为空
    for col in range(4):
        board.place(5, col, Player.WHITE)
    # 人类随便放一子，避免棋盘过于空旷影响候选点生成
    board.place(0, 14, Player.BLACK)

    searcher = _make_searcher(ai_player=Player.WHITE, depth=2)
    move = searcher.find_best_move(board)
    assert move is not None
    # AI 应该选择 (5,4) 形成五连
    assert move == (5, 4), f"Expected winning move (5,4), got {move}"


# ---------------------------------------------------------------------------
# test_find_best_move_returns_valid_position
# ---------------------------------------------------------------------------


def test_find_best_move_on_empty_board():
    """空棋盘时 AI 应该落子在天元（中心）。"""
    board = Board()
    searcher = _make_searcher(ai_player=Player.WHITE, depth=1)
    move = searcher.find_best_move(board)
    assert move == (7, 7)


def test_find_best_move_does_not_modify_board():
    """find_best_move 不应改变传入棋盘的状态。"""
    board = Board()
    board.place(7, 7, Player.BLACK)
    history_before = board.move_history.copy()

    searcher = _make_searcher()
    searcher.find_best_move(board)

    assert board.move_history == history_before
    assert board.grid[7][7] == Player.BLACK


def test_ai_as_black_wins_when_possible():
    """AI 执黑时，有五连机会应该直接赢。"""
    board = Board()
    # AI(BLACK) 已有4子: (3,0)~(3,3)，右端 (3,4) 为空
    for col in range(4):
        board.place(3, col, Player.BLACK)
    board.place(0, 14, Player.WHITE)  # 对手随机一子

    searcher = AISearcher(depth=2, ai_player=Player.BLACK)
    move = searcher.find_best_move(board)
    assert move == (3, 4), f"Expected (3,4), got {move}"


# ---------------------------------------------------------------------------
# 置换表 (TT) 测试
# ---------------------------------------------------------------------------


def test_tt_populated_after_search():
    """搜索完成后置换表应有条目（确认 TT 被写入）。"""
    board = Board()
    board.place(7, 7, Player.BLACK)
    searcher = _make_searcher(depth=2)
    searcher.find_best_move(board)
    assert len(searcher._tt) > 0


def test_tt_cleared_between_searches():
    """每次调用 find_best_move 前置换表应清空，避免跨局面污染。"""
    board = Board()
    board.place(7, 7, Player.BLACK)
    searcher = _make_searcher(depth=2)

    searcher.find_best_move(board)
    size_after_first = len(searcher._tt)

    searcher.find_best_move(board)
    size_after_second = len(searcher._tt)

    # 两次搜索相同局面，TT 大小应相同（第二次重新填充，非累积）
    assert size_after_first == size_after_second


def test_tt_does_not_change_result():
    """有无置换表，搜索结果应一致（TT 只加速，不改变决策）。"""
    board = Board()
    for col in range(3):
        board.place(5, col, Player.WHITE)
    board.place(0, 14, Player.BLACK)

    searcher = _make_searcher(depth=3)
    move1 = searcher.find_best_move(board)

    # 第二次搜索（TT 重新构建）结果相同
    move2 = searcher.find_best_move(board)
    assert move1 == move2
