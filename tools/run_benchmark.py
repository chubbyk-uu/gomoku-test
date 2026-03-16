"""CLI entry point for the Gomoku AI benchmark tool.

Usage:
    python tools/run_benchmark.py
    python tools/run_benchmark.py --depth-a 3 --depth-b 2 --games 20
    python tools/run_benchmark.py --depth-a 4 --depth-b 3 --games 10 --quiet
"""

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without a pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark import run_benchmark  # noqa: E402

from gomoku.ai.searcher import AISearcher  # noqa: E402
from gomoku.config import Player  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark two Gomoku AI searchers via automated self-play.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--depth-a", type=int, default=3, metavar="N", help="Search depth for Player A"
    )
    parser.add_argument(
        "--depth-b", type=int, default=2, metavar="N", help="Search depth for Player B"
    )
    parser.add_argument(
        "--games", type=int, default=20, metavar="N", help="Number of games to play"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-game output, print report only",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible color assignment and opening moves",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        metavar="PATH",
        help="Write per-game self-play records to a JSON file",
    )
    args = parser.parse_args()

    print(f"Player A  depth={args.depth_a}")
    print(f"Player B  depth={args.depth_b}")
    print(f"Games     {args.games}")
    if args.seed is not None:
        print(f"Seed      {args.seed}")
    if args.save_json:
        print(f"Save JSON {args.save_json}")
    print()

    # ai_player is overridden per-game inside run_benchmark; initial value is a placeholder
    player_a = AISearcher(depth=args.depth_a, ai_player=Player.BLACK)
    player_b = AISearcher(depth=args.depth_b, ai_player=Player.WHITE)

    run_benchmark(
        player_a,
        player_b,
        num_games=args.games,
        verbose=not args.quiet,
        print_report=True,
        seed=args.seed,
        save_json=args.save_json,
    )


if __name__ == "__main__":
    main()
