"""Run the fixed opening matrix used for the current official head-to-head baseline.

The matrix order matches the current investigation workflow:
1. depth=5, A as WHITE, 25 fixed center openings
2. depth=5, A as BLACK, 25 fixed center openings
3. depth=4, A as WHITE, 25 fixed center openings
4. depth=4, A as BLACK, 25 fixed center openings
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark import _EngineWrapper  # noqa: E402

from gomoku.board import Board  # noqa: E402
from gomoku.config import BOARD_SIZE, Player  # noqa: E402


@dataclass
class GroupSpec:
    depth_a: int
    depth_b: int
    a_color: str

    @property
    def group_key(self) -> str:
        return f"d{self.depth_a}_a_{self.a_color.lower()}"


def _fixed_openings() -> list[tuple[int, int]]:
    center = BOARD_SIZE // 2
    cells: list[tuple[int, int]] = []
    for row in range(center - 2, center + 3):
        for col in range(center - 2, center + 3):
            cells.append((row, col))
    return cells


def _play_fixed_game(
    sb: _EngineWrapper,
    sw: _EngineWrapper,
    opening_move: tuple[int, int],
    label_black: str,
    label_white: str,
    max_moves: int | None,
) -> tuple[Optional[Player], int, list[dict], list[float], list[float]]:
    board = Board()
    move_records: list[dict] = []
    times_black: list[float] = []
    times_white: list[float] = []

    board.place(opening_move[0], opening_move[1], Player.BLACK)
    move_records.append(
        {
            "move_no": 1,
            "engine": label_black,
            "player": Player.BLACK.name,
            "row": opening_move[0],
            "col": opening_move[1],
            "opening_random": False,
            "opening_fixed": True,
            "elapsed_ms": 0.0,
            "stats": None,
            "trace": None,
        }
    )
    num_moves = 1
    current = Player.WHITE

    while True:
        engine = sw if current == Player.WHITE else sb
        t0 = time.perf_counter()
        move = engine.find_best_move(board)
        elapsed_s = time.perf_counter() - t0
        if current == Player.BLACK:
            times_black.append(elapsed_s)
            label = label_black
        else:
            times_white.append(elapsed_s)
            label = label_white

        if move is None:
            return None, num_moves, move_records, times_black, times_white

        row, col = move
        board.place(row, col, current)
        move_records.append(
            {
                "move_no": num_moves + 1,
                "engine": label,
                "player": current.name,
                "row": row,
                "col": col,
                "opening_random": False,
                "opening_fixed": num_moves == 0,
                "elapsed_ms": round(elapsed_s * 1000, 3),
                "stats": engine.last_search_stats.__dict__,
                "trace": engine.last_decision_trace,
            }
        )
        num_moves += 1

        if board.check_win(row, col):
            return current, num_moves, move_records, times_black, times_white
        if max_moves is not None and num_moves >= max_moves:
            return None, num_moves, move_records, times_black, times_white
        if board.is_full():
            return None, num_moves, move_records, times_black, times_white

        current = Player.WHITE if current == Player.BLACK else Player.BLACK


def _write_checkpoint(
    output_path: Path,
    payload: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed opening matrix benchmark used for official baseline refreshes."
    )
    parser.add_argument("--repo-b", type=Path, required=True, help="Repo path for zhou engine")
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Checkpoint JSON written after every completed game",
    )
    parser.add_argument("--max-moves", type=int, default=120)
    parser.add_argument(
        "--group",
        choices=("d5_a_white", "d5_a_black", "d4_a_white", "d4_a_black"),
        default=None,
        help="Optional single-group run instead of the default 100-game matrix",
    )
    parser.add_argument(
        "--opening-start",
        type=int,
        default=0,
        help="Inclusive opening index start for sliced runs",
    )
    parser.add_argument(
        "--opening-count",
        type=int,
        default=None,
        help="Optional number of openings to run from opening-start",
    )
    parser.add_argument(
        "--limit-games",
        type=int,
        default=None,
        help="Optional smoke-test limit; stop after N completed games",
    )
    args = parser.parse_args()

    repo_a = str(Path.cwd().resolve())
    repo_b = str(args.repo_b.resolve())
    openings = _fixed_openings()
    opening_end = None if args.opening_count is None else args.opening_start + args.opening_count
    openings = openings[args.opening_start:opening_end]
    groups = [
        GroupSpec(depth_a=5, depth_b=5, a_color="WHITE"),
        GroupSpec(depth_a=5, depth_b=5, a_color="BLACK"),
        GroupSpec(depth_a=4, depth_b=4, a_color="WHITE"),
        GroupSpec(depth_a=4, depth_b=4, a_color="BLACK"),
    ]
    if args.group is not None:
        groups = [group for group in groups if group.group_key == args.group]

    payload: dict = {
        "repo_a": repo_a,
        "repo_b": repo_b,
        "openings": [list(move) for move in openings],
        "groups": [],
        "games": [],
    }

    game_index = 0
    for group in groups:
        group_key = group.group_key
        group_summary = {
            "group_key": group_key,
            "depth_a": group.depth_a,
            "depth_b": group.depth_b,
            "a_color": group.a_color,
            "wins_a": 0,
            "wins_b": 0,
            "draws": 0,
            "completed_games": 0,
        }
        payload["groups"].append(group_summary)

        for opening in openings:
            a_is_black = group.a_color == "BLACK"
            if a_is_black:
                sb = _EngineWrapper(group.depth_a, Player.BLACK, repo_a)
                sw = _EngineWrapper(group.depth_b, Player.WHITE, repo_b)
            else:
                sb = _EngineWrapper(group.depth_b, Player.BLACK, repo_b)
                sw = _EngineWrapper(group.depth_a, Player.WHITE, repo_a)

            try:
                winner, num_moves, move_records, times_black, times_white = _play_fixed_game(
                    sb,
                    sw,
                    opening,
                    label_black="A" if a_is_black else "B",
                    label_white="B" if a_is_black else "A",
                    max_moves=args.max_moves,
                )
            finally:
                sb.close()
                sw.close()

            if a_is_black:
                winner_is_a = winner == Player.BLACK
                winner_is_b = winner == Player.WHITE
                times_a = times_black
                times_b = times_white
            else:
                winner_is_a = winner == Player.WHITE
                winner_is_b = winner == Player.BLACK
                times_a = times_white
                times_b = times_black

            if winner_is_a:
                group_summary["wins_a"] += 1
                outcome = "A"
            elif winner_is_b:
                group_summary["wins_b"] += 1
                outcome = "B"
            else:
                group_summary["draws"] += 1
                outcome = "DRAW"

            game_index += 1
            group_summary["completed_games"] += 1
            payload["games"].append(
                {
                    "game_index": game_index,
                    "group_key": group_key,
                    "depth_a": group.depth_a,
                    "depth_b": group.depth_b,
                    "a_color": group.a_color,
                    "black_engine": "A" if a_is_black else "B",
                    "white_engine": "B" if a_is_black else "A",
                    "opening_black": list(opening),
                    "winner": winner.name if winner is not None else "DRAW",
                    "winner_engine": outcome,
                    "num_moves": num_moves,
                    "avg_ms_a": round((sum(times_a) / len(times_a) * 1000) if times_a else 0.0, 3),
                    "avg_ms_b": round((sum(times_b) / len(times_b) * 1000) if times_b else 0.0, 3),
                    "moves": move_records,
                }
            )
            _write_checkpoint(args.output_json, payload)
            print(
                f"[{game_index}/100] {group_key} opening={opening} winner={outcome} moves={num_moves}"
                f" avg_ms_a={payload['games'][-1]['avg_ms_a']:.1f}"
                f" avg_ms_b={payload['games'][-1]['avg_ms_b']:.1f}",
                flush=True,
            )
            if args.limit_games is not None and game_index >= args.limit_games:
                return


if __name__ == "__main__":
    main()
