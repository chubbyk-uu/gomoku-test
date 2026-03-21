"""Run the official fixed-opening head-to-head benchmark.

Supported opening sets:
- 5-point: center + four inner corners
- 9-point: center + four inner corners + four outer corners

The tool supports three modes:
- both: run A as WHITE and A as BLACK
- white: run only A as WHITE
- black: run only A as BLACK

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
from repo_paths import DEFAULT_OPPONENT_REPO, REPO_ROOT  # noqa: E402

from gomoku.board import Board  # noqa: E402
from gomoku.config import Player  # noqa: E402

_DEFAULT_MAX_MOVES = 120
_DEFAULT_PARALLEL = 10
_FIXED_OPENINGS_5: list[tuple[int, int]] = [
    (7, 7),
    (4, 4),
    (4, 10),
    (10, 4),
    (10, 10),
]
_FIXED_OPENINGS_9: list[tuple[int, int]] = [
    (2, 2),
    (2, 12),
    (12, 2),
    (12, 12),
    (4, 4),
    (10, 4),
    (4, 10),
    (10, 10),
    (7, 7),
]
_FIXED_OPENING_SETS: dict[str, list[tuple[int, int]]] = {
    "5": _FIXED_OPENINGS_5,
    "9": _FIXED_OPENINGS_9,
}
_FIXED_OPENINGS: list[tuple[int, int]] = _FIXED_OPENINGS_5


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


def _run_groups(
    *,
    repo_a: str,
    repo_b: str,
    openings: list[tuple[int, int]],
    groups: list[tuple[GroupSpec, Path]],
    max_moves: int,
    parallel: int,
    slices_dir: Path,
) -> None:
    tasks: list[tuple[TaskSpec, Path]] = []
    games_by_group: dict[str, list[dict]] = {}
    output_by_group: dict[str, Path] = {}
    group_spec_by_key: dict[str, GroupSpec] = {}
    slice_paths_by_group: dict[str, list[Path]] = {}

    for group, output_path in groups:
        games_by_group[group.group_key] = []
        output_by_group[group.group_key] = output_path
        group_spec_by_key[group.group_key] = group
        slice_paths_by_group[group.group_key] = []
        for index, opening in enumerate(openings):
            tasks.append((TaskSpec(group=group, opening_index=index, opening_move=opening), output_path))

    with ProcessPoolExecutor(max_workers=parallel) as executor:
        future_map = {
            executor.submit(_run_task, task, repo_a, repo_b, max_moves): (task, output_path)
            for task, output_path in tasks
        }
        for future in as_completed(future_map):
            task, output_path = future_map[future]
            result = future.result()
            group_key = task.group.group_key
            games_by_group[group_key].append(result)
            slice_payload = _group_payload(
                repo_a=repo_a,
                repo_b=repo_b,
                openings=[task.opening_move],
                group=task.group,
                games=[result],
            )
            slice_path = _slice_path(output_path, task.slice_key, slices_dir)
            _write_json(slice_path, slice_payload)
            slice_paths_by_group[group_key].append(slice_path)
            print(
                f"[{group_key}] opening={task.opening_move} winner={result['winner_engine']}"
                f" moves={result['num_moves']} avg_ms_a={result['avg_ms_a']:.1f}"
                f" avg_ms_b={result['avg_ms_b']:.1f}",
                flush=True,
            )

    for group_key, output_path in output_by_group.items():
        group = group_spec_by_key[group_key]
        final_payload = _group_payload(
            repo_a=repo_a,
            repo_b=repo_b,
            openings=openings,
            group=group,
            games=games_by_group[group_key],
        )
        _write_json(output_path, final_payload)
        for slice_path in slice_paths_by_group[group_key]:
            slice_path.unlink(missing_ok=True)

    try:
        slices_dir.rmdir()
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the official fixed-opening benchmark and write merged white/black outputs."
    )
    parser.add_argument(
        "--repo-b",
        type=Path,
        default=DEFAULT_OPPONENT_REPO,
        help="Repo path for zhou engine (default: ./opponent/zhou)",
    )
    parser.add_argument(
        "--colors",
        choices=("both", "white", "black"),
        default="both",
        help="Run both groups or only one color side for A",
    )
    parser.add_argument(
        "--opening-set",
        choices=tuple(_FIXED_OPENING_SETS),
        default="5",
        help="Choose the fixed opening set to run",
    )
    parser.add_argument(
        "--output-white-json",
        type=Path,
        default=None,
        help="Final merged JSON for the A-as-WHITE run",
    )
    parser.add_argument(
        "--output-black-json",
        type=Path,
        default=None,
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

    repo_a = str(REPO_ROOT)
    repo_b = str(args.repo_b.resolve())
    openings = list(_FIXED_OPENING_SETS[args.opening_set])
    parallel = max(1, min(args.parallel, len(openings) * 2))
    white_group = GroupSpec(depth_a=args.depth_a, depth_b=args.depth_b, a_color="WHITE")
    black_group = GroupSpec(depth_a=args.depth_a, depth_b=args.depth_b, a_color="BLACK")

    if args.colors in {"both", "white"} and args.output_white_json is None:
        parser.error("--output-white-json is required when --colors includes white")
    if args.colors in {"both", "black"} and args.output_black_json is None:
        parser.error("--output-black-json is required when --colors includes black")

    slice_anchor = (
        args.output_white_json
        if args.output_white_json is not None
        else args.output_black_json
    )
    assert slice_anchor is not None
    default_slices_dir = slice_anchor.resolve().parent / ".opening_matrix_slices"
    slices_dir = args.slices_dir.resolve() if args.slices_dir is not None else default_slices_dir
    slices_dir.mkdir(parents=True, exist_ok=True)

    groups: list[tuple[GroupSpec, Path]] = []
    if args.colors in {"both", "white"}:
        assert args.output_white_json is not None
        groups.append((white_group, args.output_white_json.resolve()))
    if args.colors in {"both", "black"}:
        assert args.output_black_json is not None
        groups.append((black_group, args.output_black_json.resolve()))

    _run_groups(
        repo_a=repo_a,
        repo_b=repo_b,
        openings=openings,
        groups=groups,
        max_moves=args.max_moves,
        parallel=parallel,
        slices_dir=slices_dir,
    )


if __name__ == "__main__":
    main()
