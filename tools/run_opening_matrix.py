"""Run the official fixed-opening head-to-head benchmark.

Current matrix:
- center: (7, 7)
- top-left: (4, 4)
- top-right: (4, 10)
- bottom-left: (10, 4)
- bottom-right: (10, 10)

The tool always runs two groups:
- A as WHITE
- A as BLACK

Each opening is run independently, slices are checkpointed while running, and the
final output is merged into two user-chosen JSON files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark import _EngineWrapper  # noqa: E402

from gomoku.board import Board  # noqa: E402
from gomoku.config import Player  # noqa: E402

_DEFAULT_MAX_MOVES = 120
_DEFAULT_PARALLEL = 10
_FIXED_OPENINGS: list[tuple[int, int]] = [
    (7, 7),
    (4, 4),
    (4, 10),
    (10, 4),
    (10, 10),
]


@dataclass(frozen=True)
class GroupSpec:
    depth_a: int
    depth_b: int
    a_color: str

    @property
    def group_key(self) -> str:
        return f"d{self.depth_a}_a_{self.a_color.lower()}"


@dataclass(frozen=True)
class TaskSpec:
    group: GroupSpec
    opening_index: int
    opening_move: tuple[int, int]

    @property
    def slice_key(self) -> str:
        row, col = self.opening_move
        return f"{self.group.group_key}_opening_{self.opening_index}_{row}_{col}"


def _play_fixed_game(
    sb: _EngineWrapper,
    sw: _EngineWrapper,
    opening_move: tuple[int, int],
    label_black: str,
    label_white: str,
    max_moves: int,
) -> tuple[str, int, list[dict], list[float], list[float]]:
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
            return "DRAW", num_moves, move_records, times_black, times_white

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
                "opening_fixed": False,
                "elapsed_ms": round(elapsed_s * 1000, 3),
                "stats": engine.last_search_stats.__dict__,
                "trace": engine.last_decision_trace,
            }
        )
        num_moves += 1

        if board.check_win(row, col):
            return current.name, num_moves, move_records, times_black, times_white
        if num_moves >= max_moves:
            return "DRAW", num_moves, move_records, times_black, times_white
        if board.is_full():
            return "DRAW", num_moves, move_records, times_black, times_white

        current = Player.WHITE if current == Player.BLACK else Player.BLACK


def _run_task(
    task: TaskSpec,
    repo_a: str,
    repo_b: str,
    max_moves: int,
) -> dict:
    group = task.group
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
            task.opening_move,
            label_black="A" if a_is_black else "B",
            label_white="B" if a_is_black else "A",
            max_moves=max_moves,
        )
    finally:
        sb.close()
        sw.close()

    if a_is_black:
        times_a = times_black
        times_b = times_white
        winner_engine = "A" if winner == Player.BLACK.name else "B" if winner == Player.WHITE.name else "DRAW"
    else:
        times_a = times_white
        times_b = times_black
        winner_engine = "A" if winner == Player.WHITE.name else "B" if winner == Player.BLACK.name else "DRAW"

    return {
        "group_key": group.group_key,
        "depth_a": group.depth_a,
        "depth_b": group.depth_b,
        "a_color": group.a_color,
        "opening_index": task.opening_index,
        "opening_black": list(task.opening_move),
        "winner": winner,
        "winner_engine": winner_engine,
        "num_moves": num_moves,
        "avg_ms_a": round((sum(times_a) / len(times_a) * 1000) if times_a else 0.0, 3),
        "avg_ms_b": round((sum(times_b) / len(times_b) * 1000) if times_b else 0.0, 3),
        "black_engine": "A" if a_is_black else "B",
        "white_engine": "B" if a_is_black else "A",
        "moves": move_records,
        "slice_key": task.slice_key,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _slice_path(final_output: Path, slice_key: str, slices_dir: Path) -> Path:
    return slices_dir / f"{final_output.stem}_{slice_key}.json"


def _group_payload(
    *,
    repo_a: str,
    repo_b: str,
    openings: list[tuple[int, int]],
    group: GroupSpec,
    games: list[dict],
) -> dict:
    ordered_games = sorted(games, key=lambda item: item["opening_index"])
    wins_a = sum(1 for game in ordered_games if game["winner_engine"] == "A")
    wins_b = sum(1 for game in ordered_games if game["winner_engine"] == "B")
    draws = sum(1 for game in ordered_games if game["winner_engine"] == "DRAW")
    return {
        "repo_a": repo_a,
        "repo_b": repo_b,
        "openings": [list(move) for move in openings],
        "group": {
            "group_key": group.group_key,
            "depth_a": group.depth_a,
            "depth_b": group.depth_b,
            "a_color": group.a_color,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "draws": draws,
            "completed_games": len(ordered_games),
        },
        "games": ordered_games,
    }


def _run_group(
    *,
    repo_a: str,
    repo_b: str,
    openings: list[tuple[int, int]],
    group: GroupSpec,
    output_path: Path,
    max_moves: int,
    parallel: int,
    slices_dir: Path,
) -> None:
    tasks = [
        TaskSpec(group=group, opening_index=index, opening_move=opening)
        for index, opening in enumerate(openings)
    ]
    completed_games: list[dict] = []
    slice_paths: list[Path] = []

    with ProcessPoolExecutor(max_workers=parallel) as executor:
        future_map = {
            executor.submit(_run_task, task, repo_a, repo_b, max_moves): task
            for task in tasks
        }
        for future in as_completed(future_map):
            task = future_map[future]
            result = future.result()
            completed_games.append(result)
            slice_payload = _group_payload(
                repo_a=repo_a,
                repo_b=repo_b,
                openings=[task.opening_move],
                group=group,
                games=[result],
            )
            slice_path = _slice_path(output_path, task.slice_key, slices_dir)
            _write_json(slice_path, slice_payload)
            slice_paths.append(slice_path)
            print(
                f"[{group.group_key}] opening={task.opening_move} winner={result['winner_engine']}"
                f" moves={result['num_moves']} avg_ms_a={result['avg_ms_a']:.1f}"
                f" avg_ms_b={result['avg_ms_b']:.1f}",
                flush=True,
            )

    final_payload = _group_payload(
        repo_a=repo_a,
        repo_b=repo_b,
        openings=openings,
        group=group,
        games=completed_games,
    )
    _write_json(output_path, final_payload)
    for slice_path in slice_paths:
        slice_path.unlink(missing_ok=True)
    try:
        slices_dir.rmdir()
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the official 5-opening benchmark and write merged white/black outputs."
    )
    parser.add_argument("--repo-b", type=Path, required=True, help="Repo path for zhou engine")
    parser.add_argument(
        "--output-white-json",
        type=Path,
        required=True,
        help="Final merged JSON for the A-as-WHITE run",
    )
    parser.add_argument(
        "--output-black-json",
        type=Path,
        required=True,
        help="Final merged JSON for the A-as-BLACK run",
    )
    parser.add_argument("--depth-a", type=int, default=5)
    parser.add_argument("--depth-b", type=int, default=5)
    parser.add_argument("--max-moves", type=int, default=_DEFAULT_MAX_MOVES)
    parser.add_argument("--parallel", type=int, default=_DEFAULT_PARALLEL)
    parser.add_argument(
        "--slices-dir",
        type=Path,
        default=None,
        help="Optional directory for temporary slice JSON files",
    )
    args = parser.parse_args()

    repo_a = str(Path.cwd().resolve())
    repo_b = str(args.repo_b.resolve())
    openings = list(_FIXED_OPENINGS)
    parallel = max(1, min(args.parallel, len(openings) * 2))
    white_group = GroupSpec(depth_a=args.depth_a, depth_b=args.depth_b, a_color="WHITE")
    black_group = GroupSpec(depth_a=args.depth_a, depth_b=args.depth_b, a_color="BLACK")

    default_slices_dir = args.output_white_json.resolve().parent / ".opening_matrix_slices"
    slices_dir = args.slices_dir.resolve() if args.slices_dir is not None else default_slices_dir
    slices_dir.mkdir(parents=True, exist_ok=True)

    _run_group(
        repo_a=repo_a,
        repo_b=repo_b,
        openings=openings,
        group=white_group,
        output_path=args.output_white_json.resolve(),
        max_moves=args.max_moves,
        parallel=parallel,
        slices_dir=slices_dir,
    )
    _run_group(
        repo_a=repo_a,
        repo_b=repo_b,
        openings=openings,
        group=black_group,
        output_path=args.output_black_json.resolve(),
        max_moves=args.max_moves,
        parallel=parallel,
        slices_dir=slices_dir,
    )


if __name__ == "__main__":
    main()
