"""Fixed puzzle suite for Gomoku tactical and performance regression checks."""

from __future__ import annotations

import time
from dataclasses import dataclass

from gomoku.ai.searcher import SearchStats
from gomoku.board import Board
from gomoku.config import Player

Placement = tuple[int, int, Player]


@dataclass(frozen=True)
class PuzzleCase:
    """One fixed board position used for move-quality and speed regression checks."""

    name: str
    category: str
    ai_player: Player
    placements: tuple[Placement, ...]
    expected_moves: frozenset[tuple[int, int]]
    acceptable_moves: frozenset[tuple[int, int]] = frozenset()
    forbidden_moves: frozenset[tuple[int, int]] = frozenset()
    description: str = ""

    def build_board(self) -> Board:
        """Return a fresh board populated with this puzzle's stones."""
        board = Board()
        for row, col, player in self.placements:
            board.place(row, col, player)
        return board


@dataclass(frozen=True)
class PuzzleResult:
    """Result for one search on one fixed puzzle."""

    case_name: str
    category: str
    move: tuple[int, int] | None
    expected_moves: frozenset[tuple[int, int]]
    acceptable_moves: frozenset[tuple[int, int]]
    forbidden_moves: frozenset[tuple[int, int]]
    elapsed_s: float
    solved: bool
    stats: SearchStats


def default_puzzle_cases() -> list[PuzzleCase]:
    """Return the default tactical and performance regression suite."""
    return [
        PuzzleCase(
            name="win_in_one_horizontal",
            category="tactic",
            ai_player=Player.WHITE,
            placements=(
                (5, 0, Player.WHITE),
                (5, 1, Player.WHITE),
                (5, 2, Player.WHITE),
                (5, 3, Player.WHITE),
                (0, 14, Player.BLACK),
            ),
            expected_moves=frozenset({(5, 4)}),
            description="Immediate horizontal win should be taken instantly.",
        ),
        PuzzleCase(
            name="block_edge_half_four",
            category="defense",
            ai_player=Player.WHITE,
            placements=(
                (0, 0, Player.BLACK),
                (0, 1, Player.BLACK),
                (0, 2, Player.BLACK),
                (0, 3, Player.BLACK),
            ),
            expected_moves=frozenset({(0, 4)}),
            description="Single forced block on an edge half-four.",
        ),
        PuzzleCase(
            name="attack_open_four",
            category="attack",
            ai_player=Player.WHITE,
            placements=(
                (7, 4, Player.WHITE),
                (7, 5, Player.WHITE),
                (7, 6, Player.WHITE),
                (6, 6, Player.BLACK),
                (8, 8, Player.BLACK),
            ),
            expected_moves=frozenset({(7, 3), (7, 7)}),
            description="AI should proactively create an open four.",
        ),
        PuzzleCase(
            name="defend_double_open_three",
            category="defense",
            ai_player=Player.WHITE,
            placements=(
                (7, 6, Player.BLACK),
                (7, 8, Player.BLACK),
                (6, 7, Player.BLACK),
                (8, 7, Player.BLACK),
            ),
            expected_moves=frozenset({(7, 7)}),
            description="Center block against a double open three.",
        ),
        PuzzleCase(
            name="defend_open_three_plus_jump_open_three",
            category="defense",
            ai_player=Player.WHITE,
            placements=(
                (7, 5, Player.BLACK),
                (7, 7, Player.BLACK),
                (6, 6, Player.BLACK),
                (8, 6, Player.BLACK),
            ),
            expected_moves=frozenset({(7, 6)}),
            description="Must block a live-three plus jump-live-three shape.",
        ),
        PuzzleCase(
            name="attack_double_open_three",
            category="attack",
            ai_player=Player.WHITE,
            placements=(
                (7, 6, Player.WHITE),
                (7, 8, Player.WHITE),
                (6, 7, Player.WHITE),
                (8, 7, Player.WHITE),
            ),
            expected_moves=frozenset({(7, 7)}),
            description="AI should proactively create a double open three.",
        ),
        PuzzleCase(
            name="quiet_midgame_connection",
            category="strategy",
            ai_player=Player.WHITE,
            placements=(
                (7, 7, Player.WHITE),
                (7, 8, Player.BLACK),
                (8, 7, Player.WHITE),
                (8, 8, Player.BLACK),
                (6, 7, Player.WHITE),
                (9, 8, Player.BLACK),
                (7, 6, Player.WHITE),
            ),
            expected_moves=frozenset({(9, 7)}),
            description="A quieter middle-game shape used for speed regression.",
        ),
        PuzzleCase(
            name="judgment_attack_over_blocked_three",
            category="judgment",
            ai_player=Player.WHITE,
            placements=(
                (7, 4, Player.BLACK),
                (7, 5, Player.BLACK),
                (7, 6, Player.BLACK),
                (7, 7, Player.WHITE),
                (6, 9, Player.WHITE),
                (7, 9, Player.WHITE),
                (8, 9, Player.WHITE),
            ),
            expected_moves=frozenset({(9, 9)}),
            description=(
                "A blocked black three is not urgent; white should prefer the stronger attack."
            ),
        ),
        PuzzleCase(
            name="judgment_expand_attack_over_small_block",
            category="judgment",
            ai_player=Player.WHITE,
            placements=(
                (7, 5, Player.BLACK),
                (7, 6, Player.BLACK),
                (7, 7, Player.WHITE),
                (6, 9, Player.WHITE),
                (7, 9, Player.WHITE),
                (8, 9, Player.WHITE),
            ),
            expected_moves=frozenset({(9, 9)}),
            description=(
                "White should extend its own live attack instead of spending a move on a small block."
            ),
        ),
        PuzzleCase(
            name="judgment_sleep_three_not_urgent",
            category="judgment",
            ai_player=Player.WHITE,
            placements=(
                (7, 5, Player.BLACK),
                (7, 6, Player.BLACK),
                (7, 7, Player.BLACK),
                (7, 8, Player.WHITE),
                (6, 4, Player.WHITE),
                (6, 5, Player.WHITE),
                (6, 6, Player.WHITE),
            ),
            expected_moves=frozenset({(6, 7)}),
            description=(
                "A one-sided black three should not outweigh a direct attacking extension for white."
            ),
        ),
        PuzzleCase(
            name="judgment_real_game_prefer_half_four_over_loose_block",
            category="judgment",
            ai_player=Player.BLACK,
            placements=(
                (6, 6, Player.BLACK),
                (6, 9, Player.BLACK),
                (7, 6, Player.BLACK),
                (7, 7, Player.BLACK),
                (7, 8, Player.BLACK),
                (7, 9, Player.BLACK),
                (8, 5, Player.BLACK),
                (8, 8, Player.BLACK),
                (8, 9, Player.BLACK),
                (5, 5, Player.WHITE),
                (6, 5, Player.WHITE),
                (6, 7, Player.WHITE),
                (6, 8, Player.WHITE),
                (6, 10, Player.WHITE),
                (7, 5, Player.WHITE),
                (7, 10, Player.WHITE),
                (8, 7, Player.WHITE),
                (9, 9, Player.WHITE),
            ),
            expected_moves=frozenset(),
            forbidden_moves=frozenset({(4, 5)}),
            description=(
                "Real-game case: black should create a strong local threat rather than play a loose block."
            ),
        ),
    ]


def run_puzzle_suite(
    searcher_factory,
    cases: list[PuzzleCase] | None = None,
    repeat: int = 1,
) -> list[PuzzleResult]:
    """Run a fixed puzzle suite and collect per-case timing and correctness."""
    selected_cases = cases or default_puzzle_cases()
    results: list[PuzzleResult] = []

    for _ in range(repeat):
        for case in selected_cases:
            board = case.build_board()
            searcher = searcher_factory(case.ai_player)
            started_at = time.perf_counter()
            move = searcher.find_best_move(board)
            elapsed_s = time.perf_counter() - started_at
            results.append(
                PuzzleResult(
                    case_name=case.name,
                    category=case.category,
                    move=move,
                    expected_moves=case.expected_moves,
                    acceptable_moves=case.acceptable_moves,
                    forbidden_moves=case.forbidden_moves,
                    elapsed_s=elapsed_s,
                    solved=_is_puzzle_move_acceptable(case, move),
                    stats=searcher.last_search_stats,
                )
            )
    return results


def _is_puzzle_move_acceptable(case: PuzzleCase, move: tuple[int, int] | None) -> bool:
    """Return whether one move satisfies a puzzle's success criteria."""
    if move is None:
        return False
    if case.expected_moves:
        return move in case.expected_moves
    if case.acceptable_moves and move in case.acceptable_moves:
        return True
    if case.forbidden_moves:
        return move not in case.forbidden_moves
    return False


def summarize_puzzle_results(results: list[PuzzleResult]) -> dict[str, dict[str, float]]:
    """Return aggregate summary per category."""
    grouped: dict[str, list[PuzzleResult]] = {}
    for result in results:
        grouped.setdefault(result.category, []).append(result)

    summary: dict[str, dict[str, float]] = {}
    for category, items in grouped.items():
        total = len(items)
        summary[category] = {
            "count": float(total),
            "solve_rate": sum(1 for item in items if item.solved) / total,
            "avg_time_s": sum(item.elapsed_s for item in items) / total,
            "max_time_s": max(item.elapsed_s for item in items),
            "avg_nodes": sum(item.stats.nodes for item in items) / total,
        }
    return summary
