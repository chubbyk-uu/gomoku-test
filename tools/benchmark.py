"""Benchmark tool for comparing two AISearcher instances through automated self-play."""

import random
import time
from dataclasses import dataclass, field
from typing import Optional

from gomoku.ai.searcher import AISearcher, SearchStats
from gomoku.board import Board
from gomoku.config import BOARD_SIZE, Player

# 开局随机落子区域：中心 OPENING_RADIUS*2+1 正方形内随机选一格
_OPENING_RADIUS = 2  # 产生 5×5 = 25 个候选开局点


@dataclass
class BenchmarkResult:
    """Aggregate results from a benchmark run.

    Attributes:
        wins_a: Number of games won by Player A.
        wins_b: Number of games won by Player B.
        draws: Number of drawn games.
        move_times_a: Per-move wall-clock times (seconds) for Player A.
        move_times_b: Per-move wall-clock times (seconds) for Player B.
        game_lengths: Number of moves played per game.
    """

    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    move_times_a: list[float] = field(default_factory=list)
    move_times_b: list[float] = field(default_factory=list)
    game_lengths: list[int] = field(default_factory=list)
    search_stats_a: list[SearchStats] = field(default_factory=list)
    search_stats_b: list[SearchStats] = field(default_factory=list)

    def total_games(self) -> int:
        """Return total number of completed games."""
        return self.wins_a + self.wins_b + self.draws

    def win_rate_a(self) -> float:
        """Win rate of Player A (draws excluded from numerator)."""
        total = self.total_games()
        return self.wins_a / total if total else 0.0

    def win_rate_b(self) -> float:
        """Win rate of Player B (draws excluded from numerator)."""
        total = self.total_games()
        return self.wins_b / total if total else 0.0

    @property
    def avg_time_a(self) -> float:
        """Average per-move search time for Player A (seconds)."""
        return sum(self.move_times_a) / len(self.move_times_a) if self.move_times_a else 0.0

    @property
    def avg_time_b(self) -> float:
        """Average per-move search time for Player B (seconds)."""
        return sum(self.move_times_b) / len(self.move_times_b) if self.move_times_b else 0.0

    @property
    def max_time_a(self) -> float:
        """Maximum single-move search time for Player A (seconds)."""
        return max(self.move_times_a) if self.move_times_a else 0.0

    @property
    def max_time_b(self) -> float:
        """Maximum single-move search time for Player B (seconds)."""
        return max(self.move_times_b) if self.move_times_b else 0.0


@dataclass
class StatsSummary:
    """Aggregate summary for a list of SearchStats."""

    avg_nodes: float = 0.0
    avg_leaf_evals: float = 0.0
    avg_ordering_evals: float = 0.0
    avg_tt_hits: float = 0.0
    avg_tt_cutoffs: float = 0.0
    avg_beta_cutoffs: float = 0.0
    avg_alpha_cutoffs: float = 0.0
    avg_immediate_wins: float = 0.0
    avg_max_branching: float = 0.0


def _summarize_stats(stats_list: list[SearchStats]) -> StatsSummary:
    """Return per-move averages over search stats collected during benchmark."""
    if not stats_list:
        return StatsSummary()

    total = len(stats_list)
    return StatsSummary(
        avg_nodes=sum(s.nodes for s in stats_list) / total,
        avg_leaf_evals=sum(s.leaf_evals for s in stats_list) / total,
        avg_ordering_evals=sum(s.ordering_evals for s in stats_list) / total,
        avg_tt_hits=sum(s.tt_hits for s in stats_list) / total,
        avg_tt_cutoffs=sum(s.tt_cutoffs for s in stats_list) / total,
        avg_beta_cutoffs=sum(s.beta_cutoffs for s in stats_list) / total,
        avg_alpha_cutoffs=sum(s.alpha_cutoffs for s in stats_list) / total,
        avg_immediate_wins=sum(s.immediate_wins for s in stats_list) / total,
        avg_max_branching=sum(s.max_branching for s in stats_list) / total,
    )


def _random_opening_move() -> tuple[int, int]:
    """Return a random position within the center 5×5 area of the board."""
    center = BOARD_SIZE // 2
    row = center + random.randint(-_OPENING_RADIUS, _OPENING_RADIUS)
    col = center + random.randint(-_OPENING_RADIUS, _OPENING_RADIUS)
    return row, col


def _play_game(
    searcher_black: AISearcher,
    searcher_white: AISearcher,
    times_black: list[float],
    times_white: list[float],
    stats_black: list[SearchStats],
    stats_white: list[SearchStats],
) -> tuple[Optional[Player], int]:
    """Play a single game to completion.

    The first move (BLACK) is placed randomly in the center 5×5 region instead of
    asking the AI, so that repeated games diverge from the start.

    Args:
        searcher_black: Searcher playing as BLACK.
        searcher_white: Searcher playing as WHITE.
        times_black: List to append BLACK's per-move times into.
        times_white: List to append WHITE's per-move times into.

    Returns:
        (winner, num_moves): winner is Player.BLACK / Player.WHITE, or None for draw.
    """
    board = Board()
    current = Player.BLACK
    num_moves = 0

    while True:
        # 第一手随机落在中心 5×5，后续交给 AI
        if num_moves == 0:
            move = _random_opening_move()
        elif current == Player.BLACK:
            t0 = time.perf_counter()
            move = searcher_black.find_best_move(board)
            times_black.append(time.perf_counter() - t0)
            stats_black.append(searcher_black.last_search_stats)
        else:
            t0 = time.perf_counter()
            move = searcher_white.find_best_move(board)
            times_white.append(time.perf_counter() - t0)
            stats_white.append(searcher_white.last_search_stats)

        if move is None:
            return None, num_moves  # no candidates — treat as draw

        row, col = move
        board.place(row, col, current)
        num_moves += 1

        if board.check_win(row, col):
            return current, num_moves
        if board.is_full():
            return None, num_moves

        current = Player.WHITE if current == Player.BLACK else Player.BLACK


def run_benchmark(
    player_a: AISearcher,
    player_b: AISearcher,
    num_games: int = 20,
    verbose: bool = True,
    print_report: bool = True,
) -> BenchmarkResult:
    """Run automated self-play benchmark between two AISearcher instances.

    Two sources of randomness are introduced so repeated games diverge:
      1. Colors are randomly assigned each game (not fixed alternation).
      2. BLACK's first move is placed randomly in the center 5×5 region.

    New AISearcher instances are created per game to ensure clean transposition tables
    and correct ai_player assignments — the passed-in instances serve as config sources.

    Args:
        player_a: "New" / challenger searcher (used for depth config).
        player_b: "Baseline" searcher (used for depth config).
        num_games: Total number of games to play.
        verbose: Print per-game results.
        print_report: Print the final summary report.

    Returns:
        BenchmarkResult with win/draw counts and timing statistics.
    """
    result = BenchmarkResult()
    times_a: list[float] = []
    times_b: list[float] = []
    stats_a: list[SearchStats] = []
    stats_b: list[SearchStats] = []

    for game_idx in range(num_games):
        # 随机分配颜色，保证大数上均衡但每局不可预测
        a_is_black = random.random() < 0.5
        if a_is_black:
            sb = AISearcher(depth=player_a.depth, ai_player=Player.BLACK)
            sw = AISearcher(depth=player_b.depth, ai_player=Player.WHITE)
        else:
            sb = AISearcher(depth=player_b.depth, ai_player=Player.BLACK)
            sw = AISearcher(depth=player_a.depth, ai_player=Player.WHITE)

        times_black: list[float] = []
        times_white: list[float] = []
        stats_black: list[SearchStats] = []
        stats_white: list[SearchStats] = []

        winner, num_moves = _play_game(sb, sw, times_black, times_white, stats_black, stats_white)
        result.game_lengths.append(num_moves)

        if a_is_black:
            times_a.extend(times_black)
            times_b.extend(times_white)
            stats_a.extend(stats_black)
            stats_b.extend(stats_white)
            winner_is_a = winner == Player.BLACK
            winner_is_b = winner == Player.WHITE
        else:
            times_a.extend(times_white)
            times_b.extend(times_black)
            stats_a.extend(stats_white)
            stats_b.extend(stats_black)
            winner_is_a = winner == Player.WHITE
            winner_is_b = winner == Player.BLACK

        if winner_is_a:
            result.wins_a += 1
            outcome = "A wins"
        elif winner_is_b:
            result.wins_b += 1
            outcome = "B wins"
        else:
            result.draws += 1
            outcome = "Draw"

        if verbose:
            color_a = "BLK" if a_is_black else "WHT"
            print(
                f"  Game {game_idx + 1:>3}/{num_games}"
                f"  A={color_a}  moves={num_moves:>3}  {outcome}"
            )

    result.move_times_a = times_a
    result.move_times_b = times_b
    result.search_stats_a = stats_a
    result.search_stats_b = stats_b

    if print_report:
        _print_report(result, player_a.depth, player_b.depth)

    return result


def _print_report(result: BenchmarkResult, depth_a: int, depth_b: int) -> None:
    """Print a formatted summary report to stdout."""
    total = result.total_games()
    avg_len = sum(result.game_lengths) / len(result.game_lengths) if result.game_lengths else 0.0
    stats_a = _summarize_stats(result.search_stats_a)
    stats_b = _summarize_stats(result.search_stats_b)

    print()
    print("=" * 52)
    print("  BENCHMARK REPORT")
    print("=" * 52)
    print(f"  Total games      : {total}")
    print(
        f"  Player A (depth={depth_a}) : {result.wins_a:>3} wins"
        f"  ({result.win_rate_a():.1%})"
    )
    print(
        f"  Player B (depth={depth_b}) : {result.wins_b:>3} wins"
        f"  ({result.win_rate_b():.1%})"
    )
    print(f"  Draws            : {result.draws:>3}")
    print(f"  Avg game length  : {avg_len:.1f} moves")
    print()
    print(
        f"  Timing A  avg={result.avg_time_a * 1000:>7.1f} ms"
        f"  max={result.max_time_a * 1000:>7.1f} ms"
    )
    print(
        f"  Timing B  avg={result.avg_time_b * 1000:>7.1f} ms"
        f"  max={result.max_time_b * 1000:>7.1f} ms"
    )
    print()
    print(
        f"  Search A  nodes={stats_a.avg_nodes:>7.1f}"
        f"  leaf={stats_a.avg_leaf_evals:>7.1f}"
        f"  order={stats_a.avg_ordering_evals:>7.1f}"
    )
    print(
        f"  Search A  tt_hits={stats_a.avg_tt_hits:>5.1f}"
        f"  tt_cut={stats_a.avg_tt_cutoffs:>5.1f}"
        f"  beta={stats_a.avg_beta_cutoffs:>5.1f}"
        f"  alpha={stats_a.avg_alpha_cutoffs:>5.1f}"
    )
    print(
        f"  Search B  nodes={stats_b.avg_nodes:>7.1f}"
        f"  leaf={stats_b.avg_leaf_evals:>7.1f}"
        f"  order={stats_b.avg_ordering_evals:>7.1f}"
    )
    print(
        f"  Search B  tt_hits={stats_b.avg_tt_hits:>5.1f}"
        f"  tt_cut={stats_b.avg_tt_cutoffs:>5.1f}"
        f"  beta={stats_b.avg_beta_cutoffs:>5.1f}"
        f"  alpha={stats_b.avg_alpha_cutoffs:>5.1f}"
    )
    print(
        f"  Branching avg  A={stats_a.avg_max_branching:>5.1f}"
        f"  B={stats_b.avg_max_branching:>5.1f}"
        f"  wins_now A/B={stats_a.avg_immediate_wins:>4.1f}/{stats_b.avg_immediate_wins:>4.1f}"
    )
    print("=" * 52)
