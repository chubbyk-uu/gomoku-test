"""Tests for the fixed-opening matrix benchmark tool."""

import importlib
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


def test_fixed_openings_use_center_and_four_corners():
    module = importlib.import_module("run_opening_matrix")

    assert module._FIXED_OPENINGS == [
        (7, 7),
        (4, 4),
        (4, 10),
        (10, 4),
        (10, 10),
    ]


def test_run_group_merges_slices_and_deletes_temporary_files(tmp_path, monkeypatch):
    module = importlib.import_module("run_opening_matrix")

    class FakeFuture:
        def __init__(self, payload):
            self._payload = payload

        def result(self):
            return self._payload

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, task, repo_a, repo_b, max_moves):
            return FakeFuture(
                {
                    "group_key": task.group.group_key,
                    "depth_a": task.group.depth_a,
                    "depth_b": task.group.depth_b,
                    "a_color": task.group.a_color,
                    "opening_index": task.opening_index,
                    "opening_black": list(task.opening_move),
                    "winner": "WHITE" if task.opening_index % 2 == 0 else "BLACK",
                    "winner_engine": "A" if task.opening_index % 2 == 0 else "B",
                    "num_moves": 20 + task.opening_index,
                    "avg_ms_a": 1.0 + task.opening_index,
                    "avg_ms_b": 2.0 + task.opening_index,
                    "black_engine": "B",
                    "white_engine": "A",
                    "moves": [{"move_no": 1, "row": task.opening_move[0], "col": task.opening_move[1]}],
                    "slice_key": task.slice_key,
                }
            )

    monkeypatch.setattr(module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(module, "as_completed", lambda futures: list(futures))

    output_path = tmp_path / "white.json"
    slices_dir = tmp_path / "slices"
    openings = list(module._FIXED_OPENINGS)
    group = module.GroupSpec(depth_a=5, depth_b=5, a_color="WHITE")

    module._run_group(
        repo_a="/repo/a",
        repo_b="/repo/b",
        openings=openings,
        group=group,
        output_path=output_path,
        max_moves=120,
        parallel=10,
        slices_dir=slices_dir,
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["group"]["group_key"] == "d5_a_white"
    assert payload["group"]["completed_games"] == 5
    assert [tuple(game["opening_black"]) for game in payload["games"]] == openings
    assert not any(slices_dir.glob("*.json"))


def test_opening_matrix_cli_writes_two_custom_outputs(tmp_path, monkeypatch):
    module = importlib.import_module("run_opening_matrix")
    captured: list[tuple[str, str, Path]] = []

    def fake_run_group(*, repo_a, repo_b, openings, group, output_path, max_moves, parallel, slices_dir):
        captured.append((group.group_key, repo_b, output_path))
        payload = module._group_payload(
            repo_a=repo_a,
            repo_b=repo_b,
            openings=openings,
            group=group,
            games=[],
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(module, "_run_group", fake_run_group)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_opening_matrix.py",
            "--repo-b",
            str(tmp_path / "zhou"),
            "--output-white-json",
            str(tmp_path / "custom_white.json"),
            "--output-black-json",
            str(tmp_path / "custom_black.json"),
        ],
    )

    module.main()

    assert (tmp_path / "custom_white.json").exists()
    assert (tmp_path / "custom_black.json").exists()
    assert [item[0] for item in captured] == ["d5_a_white", "d5_a_black"]


def test_opening_matrix_cli_can_run_only_white(tmp_path, monkeypatch):
    module = importlib.import_module("run_opening_matrix")
    captured: list[str] = []

    def fake_run_group(*, repo_a, repo_b, openings, group, output_path, max_moves, parallel, slices_dir):
        captured.append(group.group_key)
        payload = module._group_payload(
            repo_a=repo_a,
            repo_b=repo_b,
            openings=openings,
            group=group,
            games=[],
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(module, "_run_group", fake_run_group)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_opening_matrix.py",
            "--repo-b",
            str(tmp_path / "zhou"),
            "--colors",
            "white",
            "--output-white-json",
            str(tmp_path / "white_only.json"),
        ],
    )

    module.main()

    assert (tmp_path / "white_only.json").exists()
    assert captured == ["d5_a_white"]
