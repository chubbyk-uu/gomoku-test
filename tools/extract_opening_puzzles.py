"""Extract deterministic opening branches from a benchmark record file."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def _winner_engine(game: dict) -> str:
    winner = game["winner"]
    if winner == "DRAW":
        return "DRAW"
    return game["black_engine"] if winner == "BLACK" else game["white_engine"]


def _opening_key(game: dict) -> tuple:
    prefix = game["moves"][:4]
    return (
        game["a_color"],
        tuple((move["player"], move["row"], move["col"]) for move in prefix),
    )


def _first_move(game: dict) -> tuple[int, int]:
    move = game["moves"][0]
    return (move["row"], move["col"])


def _first_source(game: dict, engine: str, wanted: set[str]) -> int | None:
    for move in game["moves"]:
        if move["engine"] != engine:
            continue
        trace = move.get("trace") or {}
        if trace.get("source") in wanted:
            return move["move_no"]
    return None


def _first_minimax_score(game: dict, engine: str) -> int | float | None:
    for move in game["moves"]:
        if move["engine"] != engine:
            continue
        trace = move.get("trace") or {}
        if trace.get("source") == "minimax":
            return trace.get("score")
    return None


def _build_case(case_id: int, games: list[dict]) -> dict:
    sample = games[0]
    winner_counts = Counter(_winner_engine(game) for game in games)
    a_color = sample["a_color"]
    opening = _first_move(sample)
    branch_prefix = [
        {
            "move_no": move["move_no"],
            "engine": move["engine"],
            "player": move["player"],
            "row": move["row"],
            "col": move["col"],
            "trace_source": (move.get("trace") or {}).get("source"),
            "trace_score": (move.get("trace") or {}).get("score"),
        }
        for move in sample["moves"][:6]
    ]

    first_defs = [
        value
        for value in (
            _first_source(game, "A", {"vcf_block", "immediate_block"}) for game in games
        )
        if value is not None
    ]
    first_wins = [
        value
        for value in (
            _first_source(game, "A", {"vcf_win", "immediate_win"}) for game in games
        )
        if value is not None
    ]
    first_scores = [
        value
        for value in (_first_minimax_score(game, "A") for game in games)
        if value is not None
    ]

    return {
        "case_id": case_id,
        "a_color": a_color,
        "opening_move": {"row": opening[0], "col": opening[1]},
        "games": len(games),
        "winner_counts": dict(winner_counts),
        "a_win_rate": round(winner_counts["A"] / len(games), 4),
        "representative_prefix": branch_prefix,
        "first_a_defense_avg_move": round(sum(first_defs) / len(first_defs), 2) if first_defs else None,
        "first_a_defense_median_move": median(first_defs) if first_defs else None,
        "first_a_win_avg_move": round(sum(first_wins) / len(first_wins), 2) if first_wins else None,
        "first_a_win_median_move": median(first_wins) if first_wins else None,
        "first_a_minimax_score_avg": round(sum(first_scores) / len(first_scores), 2) if first_scores else None,
        "first_a_minimax_score_median": median(first_scores) if first_scores else None,
        "sample_game_index": sample["game_index"],
        "sample_num_moves": sample["num_moves"],
        "sample_winner": sample["winner"],
    }


def _write_markdown(path: Path, cases: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# Opening Puzzle Catalog")
    lines.append("")
    lines.append(f"Total cases: {len(cases)}")
    lines.append("")
    for case in cases:
        opening = case["opening_move"]
        lines.append(
            f"## Case {case['case_id']:02d} | A={case['a_color']} | "
            f"opening=({opening['row']},{opening['col']}) | "
            f"A win rate={case['a_win_rate']:.1%} | games={case['games']}"
        )
        lines.append("")
        lines.append(f"- Winner counts: {case['winner_counts']}")
        lines.append(f"- First A defense avg move: {case['first_a_defense_avg_move']}")
        lines.append(f"- First A win avg move: {case['first_a_win_avg_move']}")
        lines.append(f"- First A minimax score avg: {case['first_a_minimax_score_avg']}")
        lines.append("- Representative prefix:")
        for move in case["representative_prefix"]:
            lines.append(
                f"  - #{move['move_no']} {move['engine']}/{move['player']} "
                f"({move['row']},{move['col']}) "
                f"src={move['trace_source']} score={move['trace_score']}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract opening puzzle cases from benchmark JSON.")
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for game in payload["games"]:
        buckets[_opening_key(game)].append(game)

    cases = [
        _build_case(idx, games)
        for idx, (_key, games) in enumerate(
            sorted(
                buckets.items(),
                key=lambda item: (
                    item[0][0] != "WHITE",
                    item[1][0]["moves"][0]["row"],
                    item[1][0]["moves"][0]["col"],
                ),
            ),
            start=1,
        )
    ]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"cases": cases}, indent=2), encoding="utf-8")
    _write_markdown(args.output_md, cases)


if __name__ == "__main__":
    main()
