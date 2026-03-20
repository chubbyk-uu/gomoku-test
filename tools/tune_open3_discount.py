"""Quick sweep: test different isolated-OPEN_THREE discount values.

Patches evaluator._calc_total at runtime, replays to move 7 once,
then runs find_best_move for each discount value.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import gomoku.ai.evaluator as _ev
from benchmark import _EngineWrapper
from gomoku.ai.searcher import AISearcher
from gomoku.board import Board
from gomoku.config import Player

REPO_A = str(Path(__file__).parent.parent.resolve())
REPO_B = str(Path("/home/jerry/python-test/gomoku/zhou").resolve())
DEPTH = 5
OPENING = (7, 5)
DISCOUNTS = [0, 200, 300, 400, 500, 600, 700, 800]


def replay_to_move7() -> Board:
    """Hard-coded from original diagnose_rerank_75.py run (no-discount baseline)."""
    sequence = [
        (7, 5, Player.BLACK),
        (5, 3, Player.WHITE),
        (6, 6, Player.BLACK),
        (6, 2, Player.WHITE),
        (8, 4, Player.BLACK),
        (9, 3, Player.WHITE),
        (6, 5, Player.BLACK),
    ]
    board = Board()
    for row, col, player in sequence:
        board.place(row, col, player)
    return board


_orig_calc_total = _ev._calc_total


def patched_calc_total(counts, discount):
    from gomoku.ai.evaluator import Shape, SHAPE_SCORE
    open_fours = counts[Shape.OPEN_FOUR]
    half_fours = counts[Shape.HALF_FOUR]
    open_threes = counts[Shape.OPEN_THREE]
    half_threes = counts[Shape.HALF_THREE]
    open_twos = counts[Shape.OPEN_TWO]

    if counts[Shape.FIVE] > 0:
        return SHAPE_SCORE[Shape.FIVE]
    if open_fours >= 1:
        return 50_000
    if half_fours >= 2:
        return 50_000
    if half_fours >= 1 and open_threes >= 1:
        return 12_000
    if open_threes >= 2:
        return 11_000
    if half_fours >= 1 and half_threes >= 1:
        return 6_000
    if open_threes >= 1 and half_threes >= 1:
        return 3_500

    total = 0
    for shape, count in counts.items():
        total += SHAPE_SCORE[shape] * count
    if half_fours >= 1 and open_threes == 0 and half_threes == 0:
        total -= 800 * half_fours
    if open_threes >= 1 and open_twos >= 1:
        total += 400 * min(open_threes, open_twos)
    if open_twos >= 2:
        total += 150 * (open_twos - 1)
    if open_threes == 1 and half_fours == 0:
        total -= discount
    return total


def run_with_discount(board: Board, discount: int) -> None:
    import functools
    _ev._calc_total = functools.partial(patched_calc_total, discount=discount)

    searcher = AISearcher(depth=DEPTH, ai_player=Player.WHITE)
    searcher.find_best_move(board)
    candidates = searcher.last_decision_trace.root_candidates or []

    # base_rank is set by rerank; for non-reranked candidates it's absent
    base_ranks = {}
    for c in candidates:
        mv = tuple(c["move"])
        br = c.get("base_rank", "—")
        score = c.get("base_score", c.get("score", "—"))
        base_ranks[mv] = (br, score)

    r42 = base_ranks.get((4, 2), ("—", "—"))
    r71 = base_ranks.get((7, 1), ("—", "—"))
    r44 = base_ranks.get((4, 4), ("—", "—"))
    chosen = searcher.last_decision_trace.move

    print(f"discount={discount:4d}  "
          f"(4,2) base#{r42[0]} score={r42[1]}  "
          f"(7,1) base#{r71[0]} score={r71[1]}  "
          f"(4,4) base#{r44[0]} score={r44[1]}  "
          f"chosen={chosen}")

    _ev._calc_total = _orig_calc_total


def main():
    print("Replaying to move 7...")
    board = replay_to_move7()
    print()
    print(f"{'discount':>8}  (4,2) rank/score    (7,1) rank/score    (4,4) rank/score    chosen")
    print("-" * 90)
    for d in DISCOUNTS:
        run_with_discount(board, d)


if __name__ == "__main__":
    main()
