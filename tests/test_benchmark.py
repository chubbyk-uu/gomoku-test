"""Tests for the self-play benchmark helpers."""

import importlib
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


def test_run_benchmark_can_save_json_records(tmp_path):
    from benchmark import run_benchmark

    from gomoku.ai.searcher import AISearcher
    from gomoku.config import Player

    output_path = tmp_path / "selfplay.json"
    player_a = AISearcher(depth=1, ai_player=Player.BLACK, time_limit_s=None)
    player_b = AISearcher(depth=1, ai_player=Player.WHITE, time_limit_s=None)

    result = run_benchmark(
        player_a,
        player_b,
        num_games=1,
        verbose=False,
        print_report=False,
        seed=7,
        save_json=str(output_path),
    )

    assert result.total_games() == 1
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["depth_a"] == 1
    assert payload["depth_b"] == 1
    assert payload["seed"] == 7
    assert len(payload["games"]) == 1
    assert payload["games"][0]["moves"]


def test_run_benchmark_supports_repo_backed_workers(tmp_path):
    from benchmark import run_benchmark

    from gomoku.ai.searcher import AISearcher
    from gomoku.config import Player

    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "selfplay_workers.json"
    player_a = AISearcher(depth=1, ai_player=Player.BLACK, time_limit_s=None)
    player_b = AISearcher(depth=1, ai_player=Player.WHITE, time_limit_s=None)

    result = run_benchmark(
        player_a,
        player_b,
        num_games=1,
        verbose=False,
        print_report=False,
        seed=3,
        save_json=str(output_path),
        repo_a=str(repo_root),
        repo_b=str(repo_root),
    )

    assert result.total_games() == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["repo_a"] == str(repo_root)
    assert payload["repo_b"] == str(repo_root)


def test_run_benchmark_can_cap_game_length(tmp_path):
    from benchmark import run_benchmark

    from gomoku.ai.searcher import AISearcher
    from gomoku.config import Player

    output_path = tmp_path / "capped.json"
    player_a = AISearcher(depth=1, ai_player=Player.BLACK, time_limit_s=None)
    player_b = AISearcher(depth=1, ai_player=Player.WHITE, time_limit_s=None)

    result = run_benchmark(
        player_a,
        player_b,
        num_games=1,
        verbose=False,
        print_report=False,
        seed=11,
        save_json=str(output_path),
        max_moves=1,
    )

    assert result.draws == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["max_moves"] == 1
    assert payload["game_lengths"] == [1]


def test_run_benchmark_can_print_progress(capsys):
    from benchmark import run_benchmark

    from gomoku.ai.searcher import AISearcher
    from gomoku.config import Player

    player_a = AISearcher(depth=1, ai_player=Player.BLACK, time_limit_s=None)
    player_b = AISearcher(depth=1, ai_player=Player.WHITE, time_limit_s=None)

    result = run_benchmark(
        player_a,
        player_b,
        num_games=1,
        verbose=False,
        progress=True,
        print_report=False,
        seed=5,
        max_moves=2,
    )

    out = capsys.readouterr().out
    assert result.total_games() == 1
    assert "[1/1]" in out
    assert "score A/B/D=" in out
    assert "last_avg_ms A=" in out
    assert "total_avg_ms A=" in out


def test_run_benchmark_cli_defaults_to_even_depths(monkeypatch):
    module = importlib.import_module("run_benchmark")
    captured: dict[str, object] = {}

    class DummySearcher:
        def __init__(self, depth, ai_player, time_limit_s=None):
            self.depth = depth
            self.ai_player = ai_player
            self.time_limit_s = time_limit_s

    def fake_run_benchmark(player_a, player_b, **kwargs):
        captured["depth_a"] = player_a.depth
        captured["depth_b"] = player_b.depth
        captured["kwargs"] = kwargs

    monkeypatch.setattr(module, "AISearcher", DummySearcher)
    monkeypatch.setattr(module, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--games", "1", "--quiet"])

    module.main()

    assert captured["depth_a"] == 4
    assert captured["depth_b"] == 4
    assert captured["kwargs"]["progress"] is False


def test_run_benchmark_cli_can_enable_progress(monkeypatch):
    module = importlib.import_module("run_benchmark")
    captured: dict[str, object] = {}

    class DummySearcher:
        def __init__(self, depth, ai_player, time_limit_s=None):
            self.depth = depth
            self.ai_player = ai_player
            self.time_limit_s = time_limit_s

    def fake_run_benchmark(player_a, player_b, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(module, "AISearcher", DummySearcher)
    monkeypatch.setattr(module, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_benchmark.py", "--games", "1", "--quiet", "--progress"],
    )

    module.main()

    assert captured["kwargs"]["progress"] is True


def test_run_puzzle_benchmark_cli_defaults_to_even_depth(monkeypatch, capsys):
    module = importlib.import_module("run_puzzle_benchmark")
    captured: dict[str, object] = {}

    class DummySearcher:
        def __init__(self, depth, ai_player, time_limit_s=None):
            self.depth = depth
            self.ai_player = ai_player
            self.time_limit_s = time_limit_s

    class DummyResult:
        case_name = "dummy"
        solved = True
        expected_moves = {(7, 7)}
        acceptable_moves = set()
        forbidden_moves = set()
        move = (7, 7)
        elapsed_s = 0.0
        stats = type("Stats", (), {"nodes": 1})()

    monkeypatch.setattr(module, "AISearcher", DummySearcher)
    monkeypatch.setattr(module, "default_puzzle_cases", lambda: [])

    def fake_run_puzzle_suite(make_searcher, cases, repeat):
        captured["depth"] = make_searcher(module.Player.BLACK).depth
        captured["repeat"] = repeat
        return [DummyResult()]

    monkeypatch.setattr(module, "run_puzzle_suite", fake_run_puzzle_suite)
    monkeypatch.setattr(module, "summarize_puzzle_results", lambda results: {})
    monkeypatch.setattr(sys, "argv", ["run_puzzle_benchmark.py"])

    module.main()
    capsys.readouterr()

    assert captured["depth"] == 4
    assert captured["repeat"] == 1
