"""Diagnose why minimax prefers (7,1) over (4,2) at move 8.

1. Static eval breakdown just after placing each candidate (1-ply)
2. Principal Variation: both sides use gomoku-test internals at depth=5,
   showing what minimax "expects" to happen (not actual zhou vs gomoku-test).
3. Compare expected vs actual.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark import _EngineWrapper
from gomoku.ai.evaluator import (
    Shape,
    SHAPE_SCORE,
    DEFENSE_WEIGHT,
    _calc_total,
    _count_shapes,
    evaluate,
)
from gomoku.ai.searcher import AISearcher
from gomoku.board import Board
from gomoku.config import Player

REPO_A = str(Path(__file__).parent.parent.resolve())
REPO_B = str(Path("/home/jerry/python-test/gomoku/zhou").resolve())
DEPTH = 5
OPENING = (7, 5)


def replay_to_move7() -> Board:
    sb = _EngineWrapper(DEPTH, Player.BLACK, REPO_B)
    sw = _EngineWrapper(DEPTH, Player.WHITE, REPO_A)
    board = Board()
    board.place(OPENING[0], OPENING[1], Player.BLACK)
    current = Player.WHITE
    move_no = 1
    try:
        while move_no < 7:
            engine = sw if current == Player.WHITE else sb
            move = engine.find_best_move(board)
            row, col = move
            move_no += 1
            board.place(row, col, current)
            current = Player.WHITE if current == Player.BLACK else Player.BLACK
    finally:
        sb.close()
        sw.close()
    return board


def eval_breakdown(board: Board, label: str) -> None:
    """Show full evaluation breakdown for current board state."""
    for player in [Player.WHITE, Player.BLACK]:
        counts = _count_shapes(board, player)
        total = _calc_total(counts)
        shapes_str = ", ".join(
            f"{s.name}×{counts[s]}"
            for s in Shape
            if counts.get(s, 0) > 0
        )
        print(f"  {player.name:5s}: score={total:7d}  shapes=[{shapes_str}]")

    white_counts = _count_shapes(board, Player.WHITE)
    black_counts = _count_shapes(board, Player.BLACK)
    w = _calc_total(white_counts)
    b = _calc_total(black_counts)
    net_white = w - int(b * DEFENSE_WEIGHT)
    net_black = b - int(w * DEFENSE_WEIGHT)
    print(f"  net(WHITE as ai): {net_white:+d}  net(BLACK as ai): {net_black:+d}")


def extract_pv(board: Board, ai_player: Player, depth: int, max_plies: int = 8) -> list[tuple]:
    """Extract principal variation by alternating internal minimax."""
    pv = []
    current = ai_player
    b = board.copy()
    for _ in range(max_plies):
        searcher = AISearcher(depth=depth, ai_player=current)
        move = searcher.find_best_move(b)
        if move is None:
            break
        score = searcher.last_decision_trace.score
        src = searcher.last_decision_trace.source
        b.place(move[0], move[1], current)
        pv.append((move, current, score, src))
        if b.check_win(move[0], move[1]):
            pv[-1] = (move, current, score, src + "+WIN")
            break
        current = Player.WHITE if current == Player.BLACK else Player.BLACK
    return pv


def analyse_candidate(board: Board, move: tuple[int, int]) -> None:
    print(f"\n{'='*65}")
    print(f"WHITE plays {move}")
    print(f"{'='*65}")

    board.place(move[0], move[1], Player.WHITE)

    print("\n[Static eval after WHITE plays]")
    eval_breakdown(board, str(move))

    print(f"\n[Principal Variation — both sides use gomoku-test depth={DEPTH}]")
    pv = extract_pv(board, Player.BLACK, DEPTH, max_plies=8)
    for i, (mv, player, score, src) in enumerate(pv, 1):
        indent = "  " * (i + 1)
        print(f"  ply {i:2d}: {player.name:5s} {mv}  score={score}  src={src}")

    board.undo()


def main() -> None:
    print("Replaying to move 7...")
    board = replay_to_move7()
    print("Ready.\n")

    print("=== Baseline: static eval BEFORE move 8 ===")
    eval_breakdown(board, "baseline")

    for candidate in [(7, 1), (4, 2)]:
        analyse_candidate(board, candidate)


if __name__ == "__main__":
    main()
