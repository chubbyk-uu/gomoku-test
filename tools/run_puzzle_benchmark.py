"""Run the fixed Gomoku puzzle suite for correctness and speed regression checks."""

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without a pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gomoku.ai.puzzles import (  # noqa: E402
    default_puzzle_cases,
    run_puzzle_suite,
    summarize_puzzle_results,
)
from gomoku.ai.searcher import AISearcher  # noqa: E402
from gomoku.config import Player  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed Gomoku puzzle suite for speed and move-quality "
            "regression checks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--depth", type=int, default=3, metavar="N", help="Search depth")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="Repeat each puzzle N times",
    )
    parser.add_argument(
        "--category",
        action="append",
        choices=["attack", "defense", "judgment", "strategy", "tactic"],
        help="Only run selected categories; can be passed multiple times",
    )
    args = parser.parse_args()

    cases = default_puzzle_cases()
    if args.category:
        allowed = set(args.category)
        cases = [case for case in cases if case.category in allowed]

    def make_searcher(ai_player: Player) -> AISearcher:
        return AISearcher(depth=args.depth, ai_player=ai_player, time_limit_s=None)

    results = run_puzzle_suite(make_searcher, cases=cases, repeat=args.repeat)
    summary = summarize_puzzle_results(results)

    print(f"Depth   {args.depth}")
    print(f"Repeat  {args.repeat}")
    print(f"Cases   {len(cases)}")
    print()

    for result in results:
        status = "OK" if result.solved else "MISS"
        if result.expected_moves:
            target_desc = f"expected={sorted(result.expected_moves)}"
        elif result.acceptable_moves or result.forbidden_moves:
            parts: list[str] = []
            if result.acceptable_moves:
                parts.append(f"acceptable={sorted(result.acceptable_moves)}")
            if result.forbidden_moves:
                parts.append(f"forbidden={sorted(result.forbidden_moves)}")
            target_desc = " ".join(parts)
        else:
            target_desc = ""
        print(
            f"{status:>4}  {result.case_name:<36}"
            f" move={result.move!s:<10}"
            f" time={result.elapsed_s:>6.3f}s"
            f" nodes={result.stats.nodes:>4}"
            f" forcing={result.stats.forcing_wins:>2}"
        )
        if target_desc:
            print(f"      {target_desc}")

    print()
    print("By category:")
    for category in sorted(summary):
        stats = summary[category]
        print(
            f"  {category:<8}"
            f" solve={stats['solve_rate'] * 100:>5.1f}%"
            f" avg_time={stats['avg_time_s']:>6.3f}s"
            f" max_time={stats['max_time_s']:>6.3f}s"
            f" avg_nodes={stats['avg_nodes']:>6.1f}"
        )

    slowest = sorted(results, key=lambda item: item.elapsed_s, reverse=True)[:5]
    print()
    print("Slowest cases:")
    for result in slowest:
        if result.expected_moves:
            target_desc = f"expected={sorted(result.expected_moves)}"
        else:
            parts: list[str] = []
            if result.acceptable_moves:
                parts.append(f"acceptable={sorted(result.acceptable_moves)}")
            if result.forbidden_moves:
                parts.append(f"forbidden={sorted(result.forbidden_moves)}")
            target_desc = " ".join(parts)
        print(
            f"  {result.case_name:<36}"
            f" {result.elapsed_s:>6.3f}s"
            f" move={result.move!s:<10}"
            f" {target_desc}"
        )


if __name__ == "__main__":
    main()
