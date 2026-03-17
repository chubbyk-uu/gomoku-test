"""Tests for AISearcher."""

import math

import gomoku.ai.searcher as searcher_module
from gomoku.ai.searcher import AISearcher
from gomoku.ai.threats import ThreatInfo, ThreatType
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


def test_immediate_win_short_circuits_ordering():
    """一步成五时，应在排序前直接返回，避免无谓的排序评估。"""
    board = Board()
    for col in range(4):
        board.place(5, col, Player.WHITE)
    board.place(0, 14, Player.BLACK)

    searcher = _make_searcher(ai_player=Player.WHITE, depth=3)
    move = searcher.find_best_move(board)

    assert move == (5, 4)
    assert searcher.last_search_stats.immediate_wins == 1
    assert searcher.last_search_stats.ordering_evals == 0
    assert searcher.last_search_stats.completed_depth == 1


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


def test_tt_reused_between_searches():
    """同一局内连续搜索时，置换表可复用但不应破坏结果。"""
    board = Board()
    board.place(7, 7, Player.BLACK)
    searcher = _make_searcher(depth=2)

    move1 = searcher.find_best_move(board)
    size_after_first = len(searcher._tt)

    move2 = searcher.find_best_move(board)
    size_after_second = len(searcher._tt)

    assert move1 == move2
    assert size_after_first > 0
    assert size_after_second >= size_after_first


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


def test_prioritize_tt_move_only_reorders_existing_candidate():
    """TT best move 只应调整搜索顺序，不应改变候选集合。"""
    searcher = _make_searcher()
    moves = [(7, 7), (7, 8), (8, 8)]

    assert searcher._prioritize_tt_move(moves, (7, 8)) == [(7, 8), (7, 7), (8, 8)]
    assert searcher._prioritize_tt_move(moves, (9, 9)) == moves


def test_prioritize_special_moves_prefers_tt_then_killers():
    """killer move 应排在普通候选前，但仍位于 TT move 之后。"""
    searcher = _make_searcher(depth=3)
    searcher._killers[0] = [(8, 8), (7, 7)]
    moves = [(7, 7), (7, 8), (8, 8), (9, 9)]

    ordered = searcher._prioritize_special_moves(moves, (7, 8), ply=0)

    assert ordered == [(7, 8), (8, 8), (7, 7), (9, 9)]


def test_find_immediate_winning_moves_returns_all_wins():
    """开放四应识别出两端的全部一步成五点。"""
    board = Board()
    for col in range(4, 8):
        board.place(7, col, Player.BLACK)

    searcher = _make_searcher(ai_player=Player.BLACK)
    moves = board.get_candidate_moves()

    assert set(searcher._find_immediate_winning_moves(board, moves, Player.BLACK)) == {
        (7, 3),
        (7, 8),
    }


def test_is_immediate_winning_move_handles_edge_blocked_four():
    """边界封堵的冲四也应识别出唯一成五点。"""
    board = Board()
    for col in range(4):
        board.place(0, col, Player.BLACK)

    searcher = _make_searcher(ai_player=Player.BLACK)

    assert searcher._is_immediate_winning_move(board, 0, 4, Player.BLACK) is True
    assert searcher._is_immediate_winning_move(board, 1, 4, Player.BLACK) is False


def test_find_immediate_winning_moves_does_not_modify_board():
    """局部一步成五检测不应修改棋盘状态。"""
    board = Board()
    for col in range(4, 8):
        board.place(7, col, Player.BLACK)

    searcher = _make_searcher(ai_player=Player.BLACK)
    history_before = board.move_history.copy()
    hash_before = board.hash
    last_move_before = board.last_move

    moves = searcher._find_immediate_winning_moves(
        board, board.get_candidate_moves(), Player.BLACK
    )

    assert set(moves) == {(7, 3), (7, 8)}
    assert board.move_history == history_before
    assert board.hash == hash_before
    assert board.last_move == last_move_before


def test_select_search_moves_restricts_to_blocks_against_immediate_loss():
    """若对手下一手可直接获胜，候选点应缩到防点集合。"""
    board = Board()
    for col in range(4):
        board.place(0, col, Player.BLACK)

    searcher = _make_searcher(ai_player=Player.WHITE)
    stats = searcher.last_search_stats
    moves = searcher._select_search_moves(
        board,
        board.get_candidate_moves(),
        Player.WHITE,
        tt_move=None,
        stats=stats,
    )

    assert moves == [(0, 4)]
    assert stats.ordering_evals == 0


def test_select_search_moves_blocks_opponent_win_before_own_double_threat():
    """对手已有一步赢点时，必须先防，不能优先走自己的双活三。"""
    board = Board()
    for row, col in ((5, 6), (7, 7), (8, 6), (8, 8)):
        board.place(row, col, Player.BLACK)
    for row, col in ((6, 7), (7, 8), (8, 9), (10, 11)):
        board.place(row, col, Player.WHITE)

    searcher = _make_searcher(ai_player=Player.BLACK, depth=5)
    move = searcher.find_best_move(board)

    assert move == (9, 10)


def test_minimax_blocks_opponent_win_before_forcing_attack():
    """对手已有一步赢点时，不应被 forcing search 抢先短路。"""
    board = Board()
    for row, col in (
        (5, 8),
        (6, 3),
        (6, 4),
        (6, 6),
        (6, 7),
        (6, 9),
        (7, 6),
        (7, 7),
        (7, 11),
        (8, 3),
        (8, 5),
        (8, 6),
        (8, 11),
        (9, 6),
        (9, 7),
        (9, 8),
    ):
        board.place(row, col, Player.BLACK)
    for row, col in (
        (4, 9),
        (5, 6),
        (6, 8),
        (7, 4),
        (7, 8),
        (7, 10),
        (8, 7),
        (8, 8),
        (8, 9),
        (8, 10),
        (9, 4),
        (9, 9),
        (10, 6),
        (10, 8),
        (10, 9),
    ):
        board.place(row, col, Player.WHITE)

    searcher = _make_searcher(ai_player=Player.WHITE, depth=5)

    assert searcher.find_best_move(board) == (6, 5)


def test_search_avoids_open_three_only_block_when_half_four_defense_exists():
    """对手已有冲四级应手时，不应被活三优先规则误导去走弱防点。"""
    board = Board()
    black = [
        (4, 10),
        (5, 10),
        (6, 12),
        (7, 7),
        (7, 10),
        (8, 5),
        (8, 8),
        (8, 9),
        (9, 6),
        (9, 7),
        (9, 8),
        (10, 8),
        (11, 8),
    ]
    white = [
        (4, 8),
        (4, 9),
        (6, 8),
        (6, 10),
        (6, 11),
        (7, 8),
        (7, 9),
        (8, 6),
        (8, 7),
        (9, 4),
        (9, 9),
        (10, 7),
        (12, 8),
    ]
    for row, col in black:
        board.place(row, col, Player.BLACK)
    for row, col in white:
        board.place(row, col, Player.WHITE)

    searcher = _make_searcher(ai_player=Player.BLACK, depth=4)
    move = searcher.find_best_move(board)

    assert move != (8, 10)


def test_select_search_moves_prioritizes_open_four(monkeypatch):
    """阶段 2 接入后，OPEN_FOUR 应直接作为最高优先级候选返回。"""
    board = Board()
    searcher = _make_searcher(ai_player=Player.BLACK)
    candidate_moves = [(7, 7), (7, 8), (8, 8)]

    def fake_classify(_board, _moves, _player, mode="both"):
        if mode == "attack":
            return [
                ThreatInfo((7, 7), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 100, 0),
                ThreatInfo(
                    (7, 8),
                    ThreatType.OPEN_FOUR,
                    ThreatType.OPEN_FOUR,
                    ThreatType.OTHER,
                    80_000,
                    0,
                ),
                ThreatInfo((8, 8), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 100, 0),
            ]
        return [
            ThreatInfo((7, 7), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 0, 10),
            ThreatInfo((7, 8), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 0, 100),
            ThreatInfo((8, 8), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 0, 10),
        ]

    monkeypatch.setattr(searcher, "_classify_moves_cached", fake_classify)

    moves = searcher._select_search_moves(
        board,
        candidate_moves,
        Player.BLACK,
        tt_move=None,
        stats=searcher.last_search_stats,
    )

    assert moves == [(7, 8)]


def test_select_search_moves_prioritizes_four_three(monkeypatch):
    """若不存在更高等级威胁，FOUR_THREE 应优先于普通候选。"""
    board = Board()
    searcher = _make_searcher(ai_player=Player.BLACK)
    candidate_moves = [(7, 7), (7, 8), (8, 8)]

    def fake_classify(_board, _moves, _player, mode="both"):
        if mode == "attack":
            return [
                ThreatInfo((7, 7), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 100, 0),
                ThreatInfo(
                    (7, 8),
                    ThreatType.FOUR_THREE,
                    ThreatType.FOUR_THREE,
                    ThreatType.OTHER,
                    10_000,
                    0,
                ),
                ThreatInfo((8, 8), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 100, 0),
            ]
        return [
            ThreatInfo((7, 7), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 0, 10),
            ThreatInfo((7, 8), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 0, 100),
            ThreatInfo((8, 8), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 0, 10),
        ]

    monkeypatch.setattr(searcher, "_classify_moves_cached", fake_classify)

    moves = searcher._select_search_moves(
        board,
        candidate_moves,
        Player.BLACK,
        tt_move=None,
        stats=searcher.last_search_stats,
    )

    assert moves == [(7, 8)]


def test_select_search_moves_prioritizes_double_open_three(monkeypatch):
    """DOUBLE_OPEN_THREE 应作为高优先级威胁候选直接返回。"""
    board = Board()
    searcher = _make_searcher(ai_player=Player.BLACK)
    candidate_moves = [(7, 7), (7, 8), (8, 8)]

    def fake_classify(_board, _moves, _player, mode="both"):
        if mode == "attack":
            return [
                ThreatInfo((7, 7), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 100, 0),
                ThreatInfo(
                    (7, 8),
                    ThreatType.DOUBLE_OPEN_THREE,
                    ThreatType.DOUBLE_OPEN_THREE,
                    ThreatType.OTHER,
                    16_000,
                    0,
                ),
                ThreatInfo((8, 8), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 100, 0),
            ]
        return [
            ThreatInfo((7, 7), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 0, 10),
            ThreatInfo((7, 8), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 0, 100),
            ThreatInfo((8, 8), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 0, 10),
        ]

    monkeypatch.setattr(searcher, "_classify_moves_cached", fake_classify)

    moves = searcher._select_search_moves(
        board,
        candidate_moves,
        Player.BLACK,
        tt_move=None,
        stats=searcher.last_search_stats,
    )

    assert moves == [(7, 8)]


def test_select_search_moves_prefers_half_four_over_blocking_double_open_three(monkeypatch):
    """对手仅有双活三时，我方的冲四级先手应允许继续抢攻。"""
    board = Board()
    searcher = _make_searcher(ai_player=Player.BLACK)
    candidate_moves = [(7, 7), (7, 8), (8, 8)]

    def fake_classify(_board, _moves, _player, mode="both"):
        if mode == "attack":
            return [
                ThreatInfo(
                    (7, 7),
                    ThreatType.HALF_FOUR,
                    ThreatType.HALF_FOUR,
                    ThreatType.OTHER,
                    30_000,
                    0,
                ),
                ThreatInfo((7, 8), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 100, 0),
                ThreatInfo((8, 8), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 100, 0),
            ]
        return [
            ThreatInfo(
                (7, 7),
                ThreatType.DOUBLE_OPEN_THREE,
                ThreatType.OTHER,
                ThreatType.DOUBLE_OPEN_THREE,
                0,
                55_000,
            ),
            ThreatInfo(
                (7, 8),
                ThreatType.DOUBLE_OPEN_THREE,
                ThreatType.OTHER,
                ThreatType.DOUBLE_OPEN_THREE,
                0,
                55_000,
            ),
            ThreatInfo((8, 8), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 0, 10),
        ]

    monkeypatch.setattr(searcher, "_classify_moves_cached", fake_classify)

    moves = searcher._select_search_moves(
        board,
        candidate_moves,
        Player.BLACK,
        tt_move=None,
        stats=searcher.last_search_stats,
    )

    assert moves == [(7, 7)]


def test_select_search_moves_prefers_half_four_over_blocking_open_three(monkeypatch):
    """对手最高威胁仅为活三时，也应先看我方冲四级先手。"""
    board = Board()
    searcher = _make_searcher(ai_player=Player.BLACK)
    candidate_moves = [(7, 7), (7, 8), (8, 8)]

    def fake_classify(_board, _moves, _player, mode="both"):
        if mode == "attack":
            return [
                ThreatInfo(
                    (7, 7),
                    ThreatType.HALF_FOUR,
                    ThreatType.HALF_FOUR,
                    ThreatType.OTHER,
                    30_000,
                    0,
                ),
                ThreatInfo((7, 8), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 100, 0),
                ThreatInfo((8, 8), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 100, 0),
            ]
        return [
            ThreatInfo(
                (7, 7),
                ThreatType.OPEN_THREE,
                ThreatType.OTHER,
                ThreatType.OPEN_THREE,
                0,
                12_000,
            ),
            ThreatInfo(
                (7, 8),
                ThreatType.OPEN_THREE,
                ThreatType.OTHER,
                ThreatType.OPEN_THREE,
                0,
                12_000,
            ),
            ThreatInfo((8, 8), ThreatType.OTHER, ThreatType.OTHER, ThreatType.OTHER, 0, 10),
        ]

    monkeypatch.setattr(searcher, "_classify_moves_cached", fake_classify)

    moves = searcher._select_search_moves(
        board,
        candidate_moves,
        Player.BLACK,
        tt_move=None,
        stats=searcher.last_search_stats,
    )

    assert moves == [(7, 7)]


def test_dynamic_cutoff_shrinks_for_critical_scores():
    """强威胁分数应触发更窄的动态截断。"""
    scored_moves = [
        (7, 7, 180_000, 100_000, 20_000),
        (7, 8, 178_000, 95_000, 20_000),
        (8, 7, 176_000, 90_000, 20_000),
        (8, 8, 174_000, 85_000, 20_000),
        (6, 7, 172_000, 80_000, 20_000),
        (6, 8, 120_000, 50_000, 20_000),
    ]

    assert AISearcher._dynamic_cutoff(scored_moves, max_candidates=15) == 5


def test_rerank_top_moves_uses_full_evaluation_for_prefix(monkeypatch):
    """两阶段排序应允许完整评估重排粗排前缀。"""
    board = Board()
    searcher = _make_searcher(ai_player=Player.BLACK)
    scored_moves = [
        (7, 7, 100, 50, 10),
        (7, 8, 90, 40, 10),
        (8, 7, 80, 30, 10),
    ]
    exact_scores = {(7, 7): 1, (7, 8): 5, (8, 7): 3}

    def fake_evaluate(current_board: Board) -> int:
        assert current_board.last_move is not None
        return exact_scores[current_board.last_move]

    monkeypatch.setattr(searcher, "_evaluate", fake_evaluate)

    reranked = searcher._rerank_top_moves(
        board,
        scored_moves,
        rerank_limit=3,
        current_player=Player.BLACK,
        stats=searcher.last_search_stats,
    )

    assert [move[:2] for move in reranked] == [(7, 8), (8, 7), (7, 7)]


def test_rerank_top_moves_keeps_suffix_order():
    """精排只应重排前缀，后缀顺序保持原粗排结果。"""
    board = Board()
    searcher = _make_searcher(ai_player=Player.BLACK)
    scored_moves = [
        (7, 7, 100, 50, 10),
        (7, 8, 90, 40, 10),
        (8, 7, 80, 30, 10),
        (8, 8, 70, 20, 10),
    ]

    reranked = searcher._rerank_top_moves(
        board,
        scored_moves,
        rerank_limit=1,
        current_player=Player.BLACK,
        stats=searcher.last_search_stats,
    )

    assert reranked == scored_moves


def test_find_forcing_move_detects_open_four_sequence():
    """开放四应被 forcing search 识别为可证明的强制赢线。"""
    board = Board()
    board.place(7, 4, Player.BLACK)
    board.place(7, 5, Player.BLACK)
    board.place(7, 6, Player.BLACK)
    board.place(6, 5, Player.WHITE)
    board.place(8, 5, Player.WHITE)

    searcher = _make_searcher(ai_player=Player.BLACK, depth=2)

    assert searcher._find_forcing_move(board, Player.BLACK) in {(7, 3), (7, 7)}


def test_find_forcing_move_returns_none_without_forcing_line():
    """普通局面不应误判为强制赢。"""
    board = Board()
    board.place(7, 7, Player.BLACK)
    board.place(8, 8, Player.WHITE)

    searcher = _make_searcher(ai_player=Player.BLACK, depth=2)

    assert searcher._find_forcing_move(board, Player.BLACK) is None


def test_forcing_search_updates_stats_when_used():
    """forcing search 命中时应记录统计并返回对应着法。"""
    board = Board()
    board.place(7, 4, Player.BLACK)
    board.place(7, 5, Player.BLACK)
    board.place(7, 6, Player.BLACK)
    board.place(6, 5, Player.WHITE)
    board.place(8, 5, Player.WHITE)

    searcher = _make_searcher(ai_player=Player.BLACK, depth=3)
    move = searcher.find_best_move(board)

    assert move in {(7, 3), (7, 7)}
    assert searcher.last_search_stats.forcing_wins >= 1


def test_forcing_search_skipped_on_shallow_depth(monkeypatch):
    """浅层节点不应触发 forcing search 短路。"""
    board = Board()
    board.place(7, 4, Player.BLACK)
    board.place(7, 5, Player.BLACK)
    board.place(7, 6, Player.BLACK)
    board.place(6, 5, Player.WHITE)
    board.place(8, 5, Player.WHITE)

    searcher = _make_searcher(ai_player=Player.BLACK, depth=2)
    calls = {"forcing": 0}
    original_find_forcing_move = searcher._find_forcing_move

    def wrapped_find_forcing_move(*args, **kwargs):
        calls["forcing"] += 1
        return original_find_forcing_move(*args, **kwargs)

    monkeypatch.setattr(searcher, "_find_forcing_move", wrapped_find_forcing_move)

    move = searcher.find_best_move(board)

    assert move in {(7, 3), (7, 7)}
    assert calls["forcing"] == 0
    assert searcher.last_search_stats.forcing_wins == 0


def test_beta_cutoff_records_killer_move(monkeypatch):
    """产生 beta 截断的着法应被记录为当前层 killer move。"""
    board = Board()
    searcher = _make_searcher(depth=2)
    board.get_candidate_moves = lambda: [(7, 7), (7, 8)]  # type: ignore[method-assign]

    monkeypatch.setattr(searcher, "_tactical_extension_moves", lambda _board, _player: [])
    monkeypatch.setattr(
        searcher,
        "_analyze_moves_for_player",
        lambda _board, moves, _player: {move: (False, 0) for move in moves},
    )
    monkeypatch.setattr(
        searcher,
        "_select_search_moves",
        lambda _board, moves, _player, _tt_move, _stats, ply=0, **kwargs: moves,
    )
    monkeypatch.setattr(
        searcher,
        "_evaluate",
        lambda current_board: 10 if current_board.last_move == (7, 7) else -10,
    )

    score, move = searcher._minimax(
        board, 1, -math.inf, 0, True, searcher.last_search_stats
    )

    assert score == 10
    assert move == (7, 7)
    assert searcher._killers[0][0] == (7, 7)


def test_search_stats_populated_after_search():
    """搜索完成后应暴露本次搜索统计。"""
    board = Board()
    board.place(7, 7, Player.BLACK)
    searcher = _make_searcher(depth=2)

    move = searcher.find_best_move(board)

    assert move is not None
    assert searcher.last_search_stats.nodes > 0
    assert searcher.last_search_stats.ordering_evals > 0
    assert searcher.last_search_stats.max_branching > 0
    assert searcher.last_search_stats.completed_depth == 2
    assert searcher.last_search_stats.timed_out is False


def test_search_stats_reset_between_searches():
    """连续两次搜索时，统计应按单次搜索重置。"""
    board = Board()
    board.place(7, 7, Player.BLACK)
    searcher = _make_searcher(depth=2)

    searcher.find_best_move(board)
    first_stats = searcher.last_search_stats

    searcher.find_best_move(board)
    second_stats = searcher.last_search_stats

    assert second_stats.nodes > 0
    assert second_stats.nodes < first_stats.nodes
    assert second_stats.tt_hits > 0


def test_eval_cache_evicts_oldest_when_limit_reached(monkeypatch):
    """评估缓存达到上限后应淘汰旧条目，而不是整表清空。"""
    monkeypatch.setattr(searcher_module, "AI_EVAL_CACHE_MAX_SIZE", 1)

    searcher = _make_searcher(depth=1)
    searcher._eval_cache[11] = 111

    board = Board()
    board.place(7, 7, Player.BLACK)
    searcher.find_best_move(board)

    assert len(searcher._eval_cache) == 1
    assert 11 not in searcher._eval_cache


def test_tt_evicts_oldest_when_limit_reached(monkeypatch):
    """置换表达到上限后应淘汰旧条目，而不是整表清空。"""
    monkeypatch.setattr(searcher_module, "AI_TT_MAX_SIZE", 1)

    searcher = _make_searcher(depth=2)
    searcher._tt[11] = (1, 1.0, "E", None)

    board = Board()
    board.place(7, 7, Player.BLACK)
    searcher.find_best_move(board)

    assert len(searcher._tt) == 1
    assert 11 not in searcher._tt


def test_classify_moves_cache_key_respects_move_subset(monkeypatch):
    """同一局面下不同候选子集不应错误复用 threat cache。"""
    searcher = _make_searcher(depth=1)
    board = Board()
    board.place(7, 7, Player.BLACK)
    board.place(7, 8, Player.WHITE)

    seen_calls: list[list[tuple[int, int]]] = []

    def fake_classify(_board, moves, _player):
        seen_calls.append(list(moves))
        return [("marker", move) for move in moves]

    monkeypatch.setattr(searcher_module, "classify_attack_moves", fake_classify)

    first = searcher._classify_moves_cached(board, [(6, 6), (6, 7)], Player.BLACK, mode="attack")
    second = searcher._classify_moves_cached(board, [(6, 6)], Player.BLACK, mode="attack")

    assert first != second
    assert seen_calls == [[(6, 6), (6, 7)], [(6, 6)]]


def test_iterative_deepening_reaches_max_depth_without_time_limit():
    """无时间限制时，应完成到配置的最大深度。"""
    board = Board()
    board.place(7, 7, Player.BLACK)
    searcher = AISearcher(depth=3, ai_player=Player.WHITE, time_limit_s=None)

    move = searcher.find_best_move(board)

    assert move is not None
    assert searcher.last_search_stats.completed_depth == 3
    assert searcher.last_search_stats.timed_out is False


def test_time_limit_returns_last_completed_iteration(monkeypatch):
    """超时时应回退到最后一层完整搜索结果，而不是半截结果。"""
    depth1_searcher = AISearcher(depth=1, ai_player=Player.WHITE)
    board = Board()
    expected_move = depth1_searcher.find_best_move(board)

    counter = {"calls": 0}

    def fake_perf_counter() -> float:
        counter["calls"] += 1
        return 0.0 if counter["calls"] <= 40 else 1.0

    monkeypatch.setattr(searcher_module.time, "perf_counter", fake_perf_counter)

    timed_searcher = AISearcher(depth=4, ai_player=Player.WHITE, time_limit_s=0.5)
    move = timed_searcher.find_best_move(board)

    assert move == expected_move
    assert timed_searcher.last_search_stats.completed_depth == 1
    assert timed_searcher.last_search_stats.timed_out is True


def test_timeout_does_not_leak_simulated_moves_to_board(monkeypatch):
    """子递归超时时，搜索中的模拟落子必须被完整回滚。"""
    board = Board()
    board.place(7, 7, Player.BLACK)
    board_before = board.move_history.copy()
    hash_before = board.hash

    searcher = AISearcher(depth=2, ai_player=Player.WHITE, time_limit_s=1.0)
    counter = {"calls": 0}

    def fake_check_timeout() -> None:
        counter["calls"] += 1
        if counter["calls"] >= 4:
            raise searcher_module.SearchTimeout()

    monkeypatch.setattr(searcher, "_check_timeout", fake_check_timeout)

    move = searcher.find_best_move(board)

    assert move is None
    assert board.move_history == board_before
    assert board.hash == hash_before
    assert board.last_move == (7, 7)


def test_leaf_tactical_extension_replaces_static_eval_on_volatile_leaf(monkeypatch):
    """不安静叶子应进入受限 quiescence，而不是直接静态评估。"""
    board = Board()
    searcher = _make_searcher(ai_player=Player.BLACK, depth=1)

    def fake_tactical_moves(current_board: Board, _player: Player) -> list[tuple[int, int]]:
        return [(7, 7), (7, 8)] if current_board.last_move is None else []

    def fake_evaluate(current_board: Board) -> int:
        if current_board.last_move is None:
            return 0
        return {(7, 7): 5, (7, 8): 11}[current_board.last_move]

    monkeypatch.setattr(searcher, "_tactical_extension_moves", fake_tactical_moves)
    monkeypatch.setattr(searcher, "_evaluate", fake_evaluate)

    score, move = searcher._minimax(
        board, 0, -math.inf, math.inf, True, searcher.last_search_stats
    )

    assert score == 11
    assert move == (7, 8)
    assert board.move_history == []
    assert board.last_move is None


def test_leaf_tactical_extension_skips_quiet_positions(monkeypatch):
    """安静叶子不应做额外延伸，应直接进入静态评估。"""
    board = Board()
    board.place(7, 7, Player.BLACK)
    searcher = _make_searcher(ai_player=Player.WHITE, depth=1)
    calls = {"extension": 0, "evaluate": 0}

    def fake_tactical_moves(_board: Board, _player: Player) -> list[tuple[int, int]]:
        calls["extension"] += 1
        return []

    def fake_evaluate(_board: Board) -> int:
        calls["evaluate"] += 1
        return 23

    monkeypatch.setattr(searcher, "_tactical_extension_moves", fake_tactical_moves)
    monkeypatch.setattr(searcher, "_evaluate", fake_evaluate)

    score, move = searcher._minimax(
        board, 0, -math.inf, math.inf, True, searcher.last_search_stats
    )

    assert score == 23
    assert move is None
    assert calls == {"extension": 1, "evaluate": 1}
    assert board.grid[7][7] == Player.BLACK


def test_quiescence_recurses_until_position_becomes_quiet(monkeypatch):
    """quiescence 应能继续跟进下一手强制应手，而不止一层。"""
    board = Board()
    searcher = _make_searcher(ai_player=Player.BLACK, depth=1)
    seen_states: list[tuple[tuple[int, int], ...]] = []

    def fake_tactical_moves(current_board: Board, _player: Player) -> list[tuple[int, int]]:
        history = tuple((row, col) for row, col, _ in current_board.move_history)
        seen_states.append(history)
        if history == ():
            return [(7, 7)]
        if history == ((7, 7),):
            return [(7, 8)]
        return []

    def fake_evaluate(current_board: Board) -> int:
        history = tuple((row, col) for row, col, _ in current_board.move_history)
        if history == ():
            return 0
        if history == ((7, 7),):
            return 3
        if history == ((7, 7), (7, 8)):
            return 17
        return -1

    monkeypatch.setattr(searcher, "_tactical_extension_moves", fake_tactical_moves)
    monkeypatch.setattr(searcher, "_evaluate", fake_evaluate)

    score, move = searcher._minimax(
        board, 0, -math.inf, math.inf, True, searcher.last_search_stats
    )

    assert score == 3
    assert move == (7, 7)
    assert () in seen_states
    assert ((7, 7),) in seen_states
