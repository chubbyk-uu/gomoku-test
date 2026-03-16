"""Tests for the fixed puzzle suite helpers."""

from gomoku.ai.puzzles import (
    PuzzleCase,
    _is_puzzle_move_acceptable,
    default_puzzle_cases,
    run_puzzle_suite,
    summarize_puzzle_results,
)
from gomoku.ai.searcher import AISearcher
from gomoku.config import Player


def test_default_puzzle_cases_build_valid_positions():
    """默认题库中的预期点都必须是当前棋盘上的空位。"""
    for case in default_puzzle_cases():
        board = case.build_board()
        for row, col in case.expected_moves:
            assert board.grid[row, col] == Player.NONE


def test_run_puzzle_suite_collects_results():
    """题库运行器应返回正确的题目数量和分类汇总。"""

    def make_searcher(ai_player: Player) -> AISearcher:
        return AISearcher(depth=2, ai_player=ai_player, time_limit_s=None)

    cases = default_puzzle_cases()[:2]
    results = run_puzzle_suite(make_searcher, cases=cases, repeat=1)
    summary = summarize_puzzle_results(results)

    assert len(results) == 2
    assert {result.case_name for result in results} == {case.name for case in cases}
    assert set(summary) == {case.category for case in cases}


def test_puzzle_accepts_reasonable_moves_and_rejects_forbidden_move():
    case = PuzzleCase(
        name="judgment_case",
        category="judgment",
        ai_player=Player.BLACK,
        placements=(),
        expected_moves=frozenset(),
        acceptable_moves=frozenset({(5, 9), (8, 10)}),
        forbidden_moves=frozenset({(4, 5)}),
    )

    assert _is_puzzle_move_acceptable(case, (5, 9)) is True
    assert _is_puzzle_move_acceptable(case, (8, 10)) is True
    assert _is_puzzle_move_acceptable(case, (4, 5)) is False
    assert _is_puzzle_move_acceptable(case, (9, 8)) is True
