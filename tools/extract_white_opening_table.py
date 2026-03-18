"""Extract a white-side opening response table from opening puzzle cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract white opening response table.")
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    white_cases = [case for case in payload["cases"] if case["a_color"] == "WHITE"]

    table = []
    lines = ["# White Opening Table", "", f"Cases: {len(white_cases)}", ""]
    for case in white_cases:
        prefix = case["representative_prefix"]
        first_black = prefix[0]
        first_white = prefix[1]
        second_black = prefix[2]
        second_white = prefix[3]
        row = {
            "case_id": case["case_id"],
            "opening_black": [first_black["row"], first_black["col"]],
            "white_move_2": [first_white["row"], first_white["col"]],
            "black_move_3": [second_black["row"], second_black["col"]],
            "white_move_4": [second_white["row"], second_white["col"]],
            "white_move_2_source": first_white["trace_source"],
            "white_move_2_score": first_white["trace_score"],
            "white_move_4_source": second_white["trace_source"],
            "white_move_4_score": second_white["trace_score"],
            "first_a_defense_avg_move": case["first_a_defense_avg_move"],
            "sample_num_moves": case["sample_num_moves"],
            "sample_winner": case["sample_winner"],
        }
        table.append(row)
        lines.append(
            f"- Case {case['case_id']:02d} | B1=({first_black['row']},{first_black['col']}) "
            f"-> W2=({first_white['row']},{first_white['col']}) score={first_white['trace_score']} "
            f"-> B3=({second_black['row']},{second_black['col']}) "
            f"-> W4=({second_white['row']},{second_white['col']}) score={second_white['trace_score']} "
            f"| first_def={case['first_a_defense_avg_move']}"
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"rows": table}, indent=2), encoding="utf-8")
    args.output_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
