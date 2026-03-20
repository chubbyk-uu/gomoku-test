"""Tests for the standalone VCF solver."""

from gomoku.ai.threats import ThreatType
from gomoku.ai.vcf import VCFSolver
from gomoku.board import Board
from gomoku.config import Player


def _vcf_white_winning_board() -> Board:
    board = Board()
    for row, col in [(7, 0), (7, 1), (7, 2), (5, 3), (6, 3), (8, 3)]:
        board.place(row, col, Player.WHITE)
    board.place(14, 14, Player.BLACK)
    return board


def _vcf_black_winning_board() -> Board:
    board = Board()
    for row, col in [(7, 0), (7, 1), (7, 2), (5, 3), (6, 3), (8, 3)]:
        board.place(row, col, Player.BLACK)
    board.place(14, 14, Player.WHITE)
    return board


def test_vcf_finds_winning_move_without_mutating_board():
    board = _vcf_white_winning_board()
    history_before = board.move_history.copy()
    hash_before = board.hash

    solver = VCFSolver()
    move = solver.find_winning_move(board, Player.WHITE, max_depth=8)

    assert move == (7, 3)
    assert board.move_history == history_before
    assert board.hash == hash_before
    assert solver.last_trace is not None
    assert solver.last_trace["mode"] == "win"
    assert solver.last_trace["selected_move"] == [7, 3]
    assert "top_level_attack_classification" in solver.last_trace
    assert "top_level_attacks" in solver.last_trace


def test_vcf_finds_blocking_move_without_mutating_board():
    board = _vcf_black_winning_board()
    history_before = board.move_history.copy()
    hash_before = board.hash

    solver = VCFSolver()
    move = solver.find_blocking_move(board, Player.WHITE, max_depth=8)

    assert move == (7, 3)
    assert board.move_history == history_before
    assert board.hash == hash_before
    assert solver.last_trace is not None
    assert solver.last_trace["mode"] == "block"
    assert solver.last_trace["selected_move"] == [7, 3]
    assert "defense_candidates" in solver.last_trace
    assert "defense_checks" in solver.last_trace
    assert solver.last_trace["defense_checks"]
    assert solver.last_trace["defense_checks"][0]["accepted"] is True


def test_vcf_returns_none_when_position_has_no_forced_four_chain():
    board = Board()
    board.place(7, 7, Player.BLACK)
    board.place(7, 8, Player.WHITE)

    solver = VCFSolver()

    assert solver.find_winning_move(board, Player.WHITE, max_depth=8) is None
    assert solver.find_blocking_move(board, Player.WHITE, max_depth=8) is None
    assert solver.last_trace is not None
    assert solver.last_trace["mode"] == "block"
    assert solver.last_trace["selected_move"] is None


def test_generate_forced_defenses_only_returns_immediate_winning_responses(monkeypatch):
    board = Board()
    solver = VCFSolver()

    monkeypatch.setattr(solver, "_find_immediate_wins", lambda *_args, **_kwargs: [(7, 3), (7, 8)])
    monkeypatch.setattr(
        solver,
        "_generate_blocking_moves",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("broad blocks should not run")),
    )

    assert solver._generate_forced_defenses(board, Player.WHITE, Player.BLACK) == [(7, 3), (7, 8)]


def test_classify_strong_attacks_orders_by_priority_and_score(monkeypatch):
    board = Board()
    solver = VCFSolver(max_candidates=8)
    monkeypatch.setattr(solver, "_prefilter_attack_moves", lambda *_args, **_kwargs: [(7, 7), (7, 8), (8, 8), (9, 9)])

    class Info:
        def __init__(self, move, attack_type, attack_score):
            self.move = move
            self.attack_type = attack_type
            self.attack_score = attack_score

    monkeypatch.setattr(
        "gomoku.ai.vcf.classify_attack_moves",
        lambda *_args, **_kwargs: [
            Info((7, 7), ThreatType.HALF_FOUR, 30_000),
            Info((7, 8), ThreatType.OPEN_FOUR, 20_000),
            Info((8, 8), ThreatType.OPEN_FOUR, 40_000),
            Info((9, 9), ThreatType.OTHER, 999_999),
        ],
    )

    ordered = solver._classify_strong_attacks(board, Player.WHITE)

    assert ordered == [
        ((8, 8), 4, 40_000),
        ((7, 8), 4, 20_000),
        ((7, 7), 2, 30_000),
    ]


def test_prefilter_attack_moves_uses_native_vcf_probes(monkeypatch):
    board = Board()
    board.place(7, 7, Player.BLACK)
    solver = VCFSolver(max_candidates=4)
    fixed_moves = [(6, 6), (6, 7), (7, 8), (8, 7), (8, 8)]

    monkeypatch.setattr(board, "get_candidate_moves", lambda: fixed_moves)

    monkeypatch.setattr(
        "gomoku.ai.vcf._vcf_move_probes_native",
        lambda _grid, _moves, _player: [
            (False, False, False, 0),
            (False, False, True, 3_000),
            (True, False, False, 200_000),
            (False, True, False, 30_000),
            (False, False, True, 1_000),
        ],
    )

    moves = solver._prefilter_attack_moves(board, Player.WHITE)

    assert moves == [(7, 8), (8, 7), (6, 7), (6, 6), (8, 8)]


def test_prefilter_attack_moves_keeps_candidate_set_without_native_probes(monkeypatch):
    board = Board()
    board.place(7, 7, Player.BLACK)
    solver = VCFSolver(max_candidates=4)
    fixed_moves = [(6, 6), (6, 7), (7, 8), (8, 7), (8, 8)]

    monkeypatch.setattr(board, "get_candidate_moves", lambda: fixed_moves)
    monkeypatch.setattr(
        "gomoku.ai.vcf._vcf_move_probes_native",
        lambda _grid, _moves, _player: [
            (False, False, False, 0),
            (False, False, True, 3_000),
            (True, False, False, 200_000),
            (False, True, False, 30_000),
            (False, False, True, 1_000),
        ],
    )

    native_moves = solver._prefilter_attack_moves(board, Player.WHITE)

    monkeypatch.setattr("gomoku.ai.vcf._vcf_move_probes_native", None)
    fallback_moves = solver._prefilter_attack_moves(board, Player.WHITE)

    assert set(native_moves) == set(fallback_moves) == set(fixed_moves)


def test_prefilter_attack_moves_matches_python_probe_order(monkeypatch):
    board = Board()
    board.place(7, 7, Player.BLACK)
    board.place(7, 8, Player.WHITE)
    solver = VCFSolver(max_candidates=8)

    native_moves = solver._prefilter_attack_moves(board, Player.WHITE)

    monkeypatch.setattr("gomoku.ai.vcf._vcf_move_probes_native", None)
    fallback_moves = solver._prefilter_attack_moves(board, Player.WHITE)

    assert native_moves == fallback_moves


def test_generate_vcf_attacks_only_keeps_moves_with_real_follow_up(monkeypatch):
    board = Board()
    solver = VCFSolver(max_candidates=4)

    monkeypatch.setattr(
        solver,
        "_classify_strong_attacks",
        lambda *_args, **_kwargs: [
            ((7, 7), 4, 40_000),
            ((7, 8), 2, 30_000),
        ],
    )

    def fake_find_immediate_wins(current_board, _player, limit=None):
        if current_board.last_move == (7, 7):
            return []
        if current_board.last_move == (7, 8):
            return [(7, 9)] if limit == 1 else [(7, 9)]
        return []

    monkeypatch.setattr(solver, "_find_immediate_wins", fake_find_immediate_wins)

    assert solver._generate_vcf_attacks(board, Player.WHITE) == [(7, 8)]


def test_generate_blocking_moves_falls_back_to_hotness_ordering(monkeypatch):
    board = Board()
    solver = VCFSolver(max_candidates=4)

    monkeypatch.setattr(solver, "_find_immediate_wins", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(board, "get_candidate_moves", lambda: [(7, 7), (7, 8), (8, 8)])

    analysis = {
        (7, 7, Player.WHITE): (False, 10),
        (7, 8, Player.WHITE): (False, 20),
        (8, 8, Player.WHITE): (False, 5),
        (7, 7, Player.BLACK): (False, 100),
        (7, 8, Player.BLACK): (False, 30),
        (8, 8, Player.BLACK): (False, 40),
    }

    monkeypatch.setattr(
        solver,
        "_analyze_move_for_player",
        lambda _board, r, c, player: analysis[(r, c, player)],
    )

    assert solver._generate_blocking_moves(board, Player.WHITE) == [(7, 7), (7, 8), (8, 8)]


def test_vcf_find_winning_move_matches_fallback_when_native_disabled(monkeypatch):
    board = _vcf_white_winning_board()
    solver = VCFSolver()

    native_move = solver.find_winning_move(board, Player.WHITE, max_depth=8)

    monkeypatch.setattr("gomoku.ai.vcf._vcf_move_probes_native", None)
    monkeypatch.setattr("gomoku.ai.vcf._analyze_moves_native", None)
    fallback_move = solver.find_winning_move(board, Player.WHITE, max_depth=8)

    assert native_move == fallback_move == (7, 3)


def test_vcf_find_blocking_move_matches_fallback_when_native_disabled(monkeypatch):
    board = _vcf_black_winning_board()
    solver = VCFSolver()

    native_move = solver.find_blocking_move(board, Player.WHITE, max_depth=8)

    monkeypatch.setattr("gomoku.ai.vcf._vcf_move_probes_native", None)
    monkeypatch.setattr("gomoku.ai.vcf._analyze_moves_native", None)
    fallback_move = solver.find_blocking_move(board, Player.WHITE, max_depth=8)

    assert native_move == fallback_move == (7, 3)


def test_vcf_block_trace_records_each_defense_check(monkeypatch):
    board = _vcf_black_winning_board()
    solver = VCFSolver()

    monkeypatch.setattr(
        solver,
        "_generate_blocking_moves",
        lambda *_args, **_kwargs: [(7, 2), (7, 3), (7, 4)],
    )

    move = solver.find_blocking_move(board, Player.WHITE, max_depth=8)

    assert move == (7, 3)
    assert solver.last_trace is not None
    checks = solver.last_trace["defense_checks"]
    assert checks == [
        {
            "move": [7, 2],
            "wins_immediately": False,
            "remaining_vcf_move": [7, 3],
            "accepted": False,
        },
        {
            "move": [7, 3],
            "wins_immediately": False,
            "remaining_vcf_move": None,
            "accepted": True,
        },
    ]
