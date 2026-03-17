"""Run standalone VCF solver benchmarks on fixed tactical positions."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running from the repo root without a pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gomoku.ai.vcf import VCFSolver  # noqa: E402
from gomoku.board import Board  # noqa: E402
from gomoku.config import Player  # noqa: E402


@dataclass(frozen=True)
class VCFBenchmarkCase:
    name: str
    mode: str
    actor: Player
    placements: tuple[tuple[int, int, Player], ...]
    expected_move: tuple[int, int] | None

    def build_board(self) -> Board:
        board = Board()
        for row, col, player in self.placements:
            board.place(row, col, player)
        return board


def default_cases() -> list[VCFBenchmarkCase]:
    base_shape = ((7, 0), (7, 1), (7, 2), (5, 3), (6, 3), (8, 3))
    return [
        VCFBenchmarkCase(
            name="white_find_win",
            mode="win",
            actor=Player.WHITE,
            placements=tuple((row, col, Player.WHITE) for row, col in base_shape)
            + ((14, 14, Player.BLACK),),
            expected_move=(7, 3),
        ),
        VCFBenchmarkCase(
            name="white_find_block",
            mode="block",
            actor=Player.WHITE,
            placements=tuple((row, col, Player.BLACK) for row, col in base_shape)
            + ((14, 14, Player.WHITE),),
            expected_move=(7, 3),
        ),
        VCFBenchmarkCase(
            name="empty_no_vcf",
            mode="win",
            actor=Player.WHITE,
            placements=(),
            expected_move=None,
        ),
    ]


def _run_case(
    solver: VCFSolver,
    case: VCFBenchmarkCase,
    depth: int,
    repeat: int,
) -> tuple[tuple[int, int] | None, dict[str, float]]:
    last_move: tuple[int, int] | None = None
    elapsed_total = 0.0
    nodes_total = 0.0
    cache_hits_total = 0.0
    attack_total = 0.0
    defense_total = 0.0
    prefilter_total = 0.0
    classify_total = 0.0
    immediate_total = 0.0
    depth_total = 0.0

    for _ in range(repeat):
        board = case.build_board()
        if case.mode == "win":
            last_move = solver.find_winning_move(board, case.actor, depth)
        else:
            last_move = solver.find_blocking_move(board, case.actor, depth)

        stats = solver.last_stats
        elapsed_total += stats.elapsed_s
        nodes_total += stats.nodes
        cache_hits_total += stats.cache_hits
        attack_total += stats.attack_candidates
        defense_total += stats.defense_candidates
        prefilter_total += stats.prefiltered_moves
        classify_total += stats.classified_moves
        immediate_total += stats.immediate_win_checks
        depth_total += stats.max_depth_reached

    denom = float(repeat)
    return last_move, {
        "avg_time_ms": elapsed_total * 1000 / denom,
        "avg_nodes": nodes_total / denom,
        "avg_cache_hits": cache_hits_total / denom,
        "avg_attack_candidates": attack_total / denom,
        "avg_defense_candidates": defense_total / denom,
        "avg_prefiltered_moves": prefilter_total / denom,
        "avg_classified_moves": classify_total / denom,
        "avg_immediate_checks": immediate_total / denom,
        "avg_depth_reached": depth_total / denom,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark standalone VCF solving on fixed positions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--depth", type=int, default=8, metavar="N", help="VCF max depth")
    parser.add_argument("--repeat", type=int, default=10, metavar="N", help="Repeat each case N times")
    parser.add_argument(
        "--mode",
        choices=("win", "block", "both"),
        default="both",
        help="Only benchmark one query mode if desired",
    )
    args = parser.parse_args()

    cases = default_cases()
    if args.mode != "both":
        cases = [case for case in cases if case.mode == args.mode]

    solver = VCFSolver()

    print(f"Depth   {args.depth}")
    print(f"Repeat  {args.repeat}")
    print(f"Cases   {len(cases)}")
    print()

    for case in cases:
        move, summary = _run_case(solver, case, args.depth, args.repeat)
        status = "OK" if move == case.expected_move else "MISS"
        print(
            f"{status:>4}  {case.name:<18}"
            f" mode={case.mode:<5}"
            f" move={move!s:<10}"
            f" avg_time={summary['avg_time_ms']:>7.3f}ms"
            f" nodes={summary['avg_nodes']:>6.1f}"
            f" cache={summary['avg_cache_hits']:>5.1f}"
        )
        print(
            f"      attack={summary['avg_attack_candidates']:>5.1f}"
            f" defense={summary['avg_defense_candidates']:>5.1f}"
            f" prefilter={summary['avg_prefiltered_moves']:>5.1f}"
            f" classify={summary['avg_classified_moves']:>5.1f}"
            f" immediate={summary['avg_immediate_checks']:>5.1f}"
            f" depth={summary['avg_depth_reached']:>4.1f}"
        )


if __name__ == "__main__":
    main()
