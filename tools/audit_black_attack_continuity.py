"""Audit black attack-continuity metrics on representative regression samples.

This script compares an old stronger black line and a newer weaker line after the
first divergent move, using two lightweight metrics:

1. white_effective_escape_count:
   Number of white stabilizers within a small score gap from the best stabilizer.
2. black_followup_width:
   Number of black follow-up moves whose local attack hotness stays within a
   small gap from the best follow-up attack score.
3. best_reply_escape_margin:
   Under the strongest white reply, how far apart the best and second-best
   white stabilizers are. Larger margins mean white has fewer near-equal escapes.
4. axis_extension_concentration:
   How strongly black's best follow-up candidates stay aligned with the current
   black attack axis after the best white stabilizer.

The script also reports an offline line-quality classifier based on the two
most informative metrics observed so far:

- lower white_effective_escape_count is better for black
- higher best_reply_escape_margin is better for black

It additionally evaluates a gated joint-classifier experiment:

- compare only candidates already promoted into a small top-K window
- allow the classifier to intervene only when a candidate is close enough to
  the current top-1 score
- current gate under test:
  - same-sign scores: candidate / top1 >= 0.95
  - otherwise: top1 - candidate <= 10
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gomoku.ai.searcher import AISearcher  # noqa: E402
from gomoku.board import Board  # noqa: E402
from gomoku.config import BOARD_SIZE, Player  # noqa: E402


SAMPLES = [
    {
        "label": "7,7",
        "opening": (7, 7),
        "old_line": [(6, 6), (5, 7), (4, 6), (6, 7), (4, 7), (5, 8)],
        "new_line": [(6, 6), (5, 7), (4, 6), (6, 7), (4, 7), (7, 8)],
    },
    {
        "label": "7,6",
        "opening": (7, 6),
        "old_line": [(6, 5), (5, 6)],
        "new_line": [(6, 5), (7, 4)],
    },
    {
        "label": "8,8",
        "opening": (8, 8),
        "old_line": [(7, 7), (6, 8), (5, 7), (7, 8), (5, 8), (6, 9)],
        "new_line": [(7, 7), (6, 8), (5, 7), (7, 8), (5, 8), (8, 9)],
    },
]


# Hand-audited true divergence points used to test a gated joint classifier.
# top1_score is the current baseline top-1 rerank at the true divergence step;
# old/new scores are the strong/weak line rerank scores at that same step.
DIVERGENCE_GATE_CASES = [
    {
        "label": "7,7@B7",
        "top1_score": 4465.9,
        "old_score": 4456.9,
        "new_score": 4465.9,
        "expected": "old_better",
    },
    {
        "label": "8,8@B7",
        "top1_score": 4465.9,
        "old_score": 4456.9,
        "new_score": 4465.9,
        "expected": "old_better",
    },
    {
        "label": "7,6@B3",
        "top1_score": 1441.13,
        "old_score": 1441.13,
        "new_score": 1441.13,
        "expected": "tie",
    },
    {
        "label": "5,8@B3",
        "top1_score": 1441.13,
        "old_score": 1441.13,
        "new_score": 1441.13,
        "expected": "tie",
    },
]


WHITE_ESCAPE_EVAL_GAP = 300.0
BLACK_FOLLOWUP_ATTACK_GAP = 1200
AXIS_DISTANCE_GAP = 2


def load_searcher_from_commit(worktree: str) -> AISearcher:
    """Load AISearcher class from another worktree without changing cwd."""
    src_root = Path(worktree) / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    spec = importlib.util.spec_from_file_location(
        "_audit_searcher_module", src_root / "gomoku" / "ai" / "searcher.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AISearcher(depth=5, ai_player=Player.BLACK)


def black_stones(board: Board) -> list[tuple[int, int]]:
    stones: list[tuple[int, int]] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board.grid[row, col] == int(Player.BLACK):
                stones.append((row, col))
    return stones


def attack_axis(board: Board) -> tuple[float, float] | None:
    stones = black_stones(board)
    if len(stones) < 2:
        return None
    best_pair: tuple[tuple[int, int], tuple[int, int]] | None = None
    best_dist = 10**9
    last = stones[-1]
    for stone in stones[:-1]:
        dist = abs(last[0] - stone[0]) + abs(last[1] - stone[1])
        if dist < best_dist:
            best_dist = dist
            best_pair = (last, stone)
    if best_pair is None:
        return None
    (r1, c1), (r2, c2) = best_pair
    dr = r1 - r2
    dc = c1 - c2
    if dr == 0 and dc == 0:
        return None
    return (float(dr), float(dc))


def axis_distance(axis: tuple[float, float], move: tuple[int, int], origin: tuple[int, int]) -> float:
    ar, ac = axis
    dr = move[0] - origin[0]
    dc = move[1] - origin[1]
    return abs(ar * dc - ac * dr)


def apply_prefix(board: Board, opening: tuple[int, int], line: list[tuple[int, int]]) -> None:
    board.place(opening[0], opening[1], Player.BLACK)
    current = Player.WHITE
    for row, col in line:
        board.place(row, col, current)
        current = Player.BLACK if current == Player.WHITE else Player.WHITE


def best_black_probe_candidate(searcher: AISearcher, board: Board) -> dict[str, object]:
    score, _move = searcher._minimax(  # pylint: disable=protected-access
        board,
        1,
        float("-inf"),
        float("inf"),
        maximizing=True,
        stats=searcher.last_search_stats.__class__(),
        root_trace=[],
    )
    root_candidates: list[dict[str, object]] = []
    _score, _move = searcher._minimax(  # pylint: disable=protected-access
        board,
        1,
        float("-inf"),
        float("inf"),
        maximizing=True,
        stats=searcher.last_search_stats.__class__(),
        root_trace=root_candidates,
    )
    chosen = max(root_candidates, key=lambda c: float(c.get("score", float("-inf"))))
    return {"score": score, "root_candidates": root_candidates, "chosen": chosen}


def find_candidate(root_candidates: list[dict[str, object]], move: tuple[int, int]) -> dict[str, object] | None:
    for candidate in root_candidates:
        mv = candidate.get("move")
        if isinstance(mv, list) and tuple(mv) == move:
            return candidate
    return None


def rerank_gap_summary(root_candidates: list[dict[str, object]]) -> dict[str, float] | None:
    reranks = [
        float(candidate.get("rerank_score", candidate.get("score", float("-inf"))))
        for candidate in root_candidates
    ]
    if len(reranks) < 2:
        return None
    reranks.sort(reverse=True)
    return {
        "top1_top2_gap": reranks[0] - reranks[1],
        "top2_top3_gap": reranks[1] - reranks[2] if len(reranks) > 2 else 0.0,
    }


def measure_metrics(searcher: AISearcher, board: Board, black_move: tuple[int, int]) -> dict[str, object]:
    board.place(black_move[0], black_move[1], Player.BLACK)
    try:
        probe = searcher._probe_black_reply_score(board, black_move)  # pylint: disable=protected-access
        escape_counts: list[int] = []
        followup_widths: list[int] = []
        best_reply_escape_margin: float | None = None
        best_reply_convergence_gap: dict[str, float] | None = None
        reply_summaries: list[dict[str, object]] = []
        for reply in probe.get("reply_candidates", []):
            reply_move = tuple(reply["move"])
            stabilizers = reply.get("stabilizer_candidates", [])
            if stabilizers:
                stab_scores = sorted((float(s["score"]) for s in stabilizers), reverse=True)
                best_stab = stab_scores[0]
                escape_count = sum(
                    1 for s in stabilizers if best_stab - float(s["score"]) <= WHITE_ESCAPE_EVAL_GAP
                )
                margin = best_stab - stab_scores[1] if len(stab_scores) > 1 else best_stab
            else:
                escape_count = 0
                margin = None
            escape_counts.append(escape_count)

            board.place(reply_move[0], reply_move[1], Player.WHITE)
            try:
                if stabilizers:
                    best_stab_move = tuple(
                        max(stabilizers, key=lambda s: float(s["score"]))["move"]
                    )
                    board.place(best_stab_move[0], best_stab_move[1], Player.BLACK)
                    try:
                        root_trace: list[dict[str, object]] = []
                        _score, _move = searcher._minimax(  # pylint: disable=protected-access
                            board,
                            1,
                            float("-inf"),
                            float("inf"),
                            maximizing=True,
                            stats=searcher.last_search_stats.__class__(),
                            root_trace=root_trace,
                        )
                        convergence_gap = rerank_gap_summary(root_trace)
                        follow_moves = searcher._candidate_moves(board)  # pylint: disable=protected-access
                        attacks = [
                            searcher._local_hotness(board, row, col, Player.BLACK)  # pylint: disable=protected-access
                            for row, col in follow_moves
                        ]
                        if attacks:
                            best_attack = max(attacks)
                            width = sum(
                                1 for attack in attacks if best_attack - attack <= BLACK_FOLLOWUP_ATTACK_GAP
                            )
                            axis = attack_axis(board)
                            if axis is not None:
                                scored = list(zip(follow_moves, attacks))
                                top_followups = [
                                    move
                                    for move, attack in scored
                                    if best_attack - attack <= BLACK_FOLLOWUP_ATTACK_GAP
                                ]
                                distances = [
                                    axis_distance(axis, move, best_stab_move) for move in top_followups
                                ]
                                axis_concentration = sum(
                                    1 for distance in distances if distance <= AXIS_DISTANCE_GAP
                                ) / len(distances)
                            else:
                                axis_concentration = 0.0
                        else:
                            width = 0
                            axis_concentration = 0.0
                    finally:
                        board.undo()
                else:
                    width = 0
                    axis_concentration = 0.0
                    convergence_gap = None
            finally:
                board.undo()

            followup_widths.append(width)
            if reply.get("score") == probe.get("max_reply_score"):
                if margin is not None and (
                    best_reply_escape_margin is None or margin > best_reply_escape_margin
                ):
                    best_reply_escape_margin = margin
                if convergence_gap is not None:
                    if (
                        best_reply_convergence_gap is None
                        or convergence_gap["top1_top2_gap"]
                        > best_reply_convergence_gap["top1_top2_gap"]
                    ):
                        best_reply_convergence_gap = convergence_gap
            reply_summaries.append(
                {
                    "reply": reply_move,
                    "reply_score": reply.get("score"),
                    "escape_count": escape_count,
                    "escape_margin": margin,
                    "followup_width": width,
                    "axis_extension_concentration": axis_concentration,
                    "attack_front_convergence": convergence_gap,
                }
            )

        return {
            "max_reply_score": probe.get("max_reply_score"),
            "avg_reply_score": probe.get("avg_reply_score"),
            "mean_escape_count": sum(escape_counts) / len(escape_counts) if escape_counts else 0.0,
            "best_reply_escape_margin": best_reply_escape_margin,
            "best_reply_convergence_gap": best_reply_convergence_gap,
            "mean_followup_width": sum(followup_widths) / len(followup_widths)
            if followup_widths
            else 0.0,
            "mean_axis_extension_concentration": (
                sum(r["axis_extension_concentration"] for r in reply_summaries) / len(reply_summaries)
                if reply_summaries
                else 0.0
            ),
            "replies": reply_summaries,
        }
    finally:
        board.undo()


def line_quality_key(metrics: dict[str, object]) -> tuple[float, float]:
    """Offline classifier key for ranking black lines.

    Lower escape count should rank ahead; larger escape margin should rank ahead.
    """
    escape_count = float(metrics.get("mean_escape_count", 99.0))
    escape_margin = float(metrics.get("best_reply_escape_margin") or 0.0)
    return (escape_count, -escape_margin)


def compare_lines(old_metrics: dict[str, object], new_metrics: dict[str, object]) -> dict[str, object]:
    old_key = line_quality_key(old_metrics)
    new_key = line_quality_key(new_metrics)
    if old_key < new_key:
        verdict = "old_better"
    elif old_key > new_key:
        verdict = "new_better"
    else:
        verdict = "tie"
    return {
        "old_key": {
            "escape_count": old_key[0],
            "neg_escape_margin": old_key[1],
        },
        "new_key": {
            "escape_count": new_key[0],
            "neg_escape_margin": new_key[1],
        },
        "verdict": verdict,
    }


def candidate_passes_gate(top1_score: float, candidate_score: float) -> dict[str, object]:
    """Apply the experimental gate under test.

    - If candidate and top1 have the same sign and top1 is non-zero, use ratio.
    - Otherwise fall back to absolute score difference.
    """
    same_sign = (
        (top1_score > 0 and candidate_score > 0)
        or (top1_score < 0 and candidate_score < 0)
        or (top1_score == 0 and candidate_score == 0)
    )
    ratio = None
    if same_sign and top1_score != 0.0:
        ratio = candidate_score / top1_score
        passed = ratio >= 0.95
        reason = "ratio>=0.95"
    else:
        diff = top1_score - candidate_score
        passed = diff <= 10.0
        reason = "diff<=10"
    return {
        "same_sign": same_sign,
        "ratio": ratio,
        "diff": top1_score - candidate_score,
        "passed": passed,
        "reason": reason,
    }


def evaluate_gate_cases(report: list[dict[str, object]]) -> list[dict[str, object]]:
    """Test whether the gated joint classifier would keep and distinguish lines."""
    metrics_by_label = {entry["label"]: entry for entry in report}
    results: list[dict[str, object]] = []
    for case in DIVERGENCE_GATE_CASES:
        label = case["label"].split("@", 1)[0]
        sample_metrics = metrics_by_label.get(label)
        if sample_metrics is None:
            continue
        old_gate = candidate_passes_gate(case["top1_score"], case["old_score"])
        new_gate = candidate_passes_gate(case["top1_score"], case["new_score"])
        classifier = sample_metrics["line_quality_classifier"]
        results.append(
            {
                "label": case["label"],
                "top1_score": case["top1_score"],
                "old_score": case["old_score"],
                "new_score": case["new_score"],
                "old_gate": old_gate,
                "new_gate": new_gate,
                "classifier_verdict": classifier["verdict"],
                "expected_verdict": case["expected"],
                "gate_keeps_both": bool(old_gate["passed"] and new_gate["passed"]),
                "classifier_matches_expected": classifier["verdict"] == case["expected"],
            }
        )
    return results


def main() -> None:
    old_searcher = load_searcher_from_commit("/tmp/gomoku_873")
    new_searcher = load_searcher_from_commit("/tmp/gomoku_fbb")
    report: list[dict[str, object]] = []
    for sample in SAMPLES:
        opening = sample["opening"]
        old_board = Board()
        apply_prefix(old_board, opening, sample["old_line"])
        new_board = Board()
        apply_prefix(new_board, opening, sample["new_line"])

        old_move = sample["old_line"][-1]
        new_move = sample["new_line"][-1]
        old_metrics = measure_metrics(old_searcher, old_board, old_move)
        new_metrics = measure_metrics(new_searcher, new_board, new_move)
        classifier = compare_lines(old_metrics, new_metrics)
        report.append(
            {
                "label": sample["label"],
                "opening": opening,
                "old_move": old_move,
                "new_move": new_move,
                "old_metrics": old_metrics,
                "new_metrics": new_metrics,
                "line_quality_classifier": classifier,
            }
        )

    gate_report = evaluate_gate_cases(report)
    print(json.dumps({"line_reports": report, "gated_joint_classifier": gate_report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
