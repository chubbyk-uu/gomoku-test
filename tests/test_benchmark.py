"""Tests for the self-play benchmark helpers."""

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
