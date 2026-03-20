"""Deep-dive: why does (4,4) always score 411 in minimax?

1. Score at each iterative depth (1..5) after placing (4,4)
2. PV: extract principal variation by alternating internal minimax
3. Leaf eval breakdown: static eval at each PV node
4. Check if any PV node has isolated OPEN_THREE
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gomoku.ai.evaluator import (
    Shape, SHAPE_SCORE, DEFENSE_WEIGHT,
    _calc_total, _count_shapes, evaluate,
)
from gomoku.ai.searcher import AISearcher
from gomoku.board import Board
from gomoku.config import Player

SEQUENCE_TO_MOVE7 = [
    (7, 5, Player.BLACK),
    (5, 3, Player.WHITE),
    (6, 6, Player.BLACK),
    (6, 2, Player.WHITE),
    (8, 4, Player.BLACK),
    (9, 3, Player.WHITE),
    (6, 5, Player.BLACK),
]


def build_board() -> Board:
    board = Board()
    for row, col, player in SEQUENCE_TO_MOVE7:
        board.place(row, col, player)
    return board


def eval_detail(board: Board) -> str:
    w = _count_shapes(board, Player.WHITE)
    b = _count_shapes(board, Player.BLACK)
    w_total = _calc_total(w)
    b_total = _calc_total(b)
    net = w_total - int(b_total * DEFENSE_WEIGHT)

    w_shapes = ", ".join(f"{s.name}×{w[s]}" for s in Shape if w.get(s, 0) > 0)
    b_shapes = ", ".join(f"{s.name}×{b[s]}" for s in Shape if b.get(s, 0) > 0)
    isolated_w = (w.get(Shape.OPEN_THREE, 0) == 1 and w.get(Shape.HALF_FOUR, 0) == 0
                  and w.get(Shape.HALF_THREE, 0) == 0)
    isolated_b = (b.get(Shape.OPEN_THREE, 0) == 1 and b.get(Shape.HALF_FOUR, 0) == 0
                  and b.get(Shape.HALF_THREE, 0) == 0)
    flags = ""
    if isolated_w:
        flags += " [W:isolated_open3]"
    if isolated_b:
        flags += " [B:isolated_open3]"

    return (f"net={net:+5d}  W={w_total}[{w_shapes}]  B={b_total}[{b_shapes}]{flags}")


def score_at_depth(board: Board, player: Player, depth: int) -> tuple[float, tuple]:
    s = AISearcher(depth=depth, ai_player=player)
    move = s.find_best_move(board)
    score = s.last_decision_trace.score
    src = s.last_decision_trace.source
    return score, move, src


def main() -> None:
    board = build_board()

    print("=== PART 1: (4,4) score at each iterative depth (white to play) ===\n")
    board.place(4, 4, Player.WHITE)
    # White just played (4,4), now it's black's turn — run black's searcher
    for d in range(1, 6):
        score, move, src = score_at_depth(board, Player.BLACK, d)
        # score is from BLACK's perspective
        print(f"  depth={d}  BLACK best={move} score={score} src={src}  "
              f"(white net = {-score if score is not None else '?'})")
    board.undo()

    print("\n=== PART 2: (4,2) score at each iterative depth (white to play) ===\n")
    board.place(4, 2, Player.WHITE)
    for d in range(1, 6):
        score, move, src = score_at_depth(board, Player.BLACK, d)
        print(f"  depth={d}  BLACK best={move} score={score} src={src}  "
              f"(white net = {-score if score is not None else '?'})")
    board.undo()

    print("\n=== PART 3: PV for (4,4) — both sides use internal minimax depth=5 ===\n")
    board.place(4, 4, Player.WHITE)
    print(f"  After W(4,4): {eval_detail(board)}")
    current = Player.BLACK
    for ply in range(1, 10):
        s = AISearcher(depth=5, ai_player=current)
        move = s.find_best_move(board)
        if move is None:
            print(f"  ply {ply}: {current.name} has no move")
            break
        score = s.last_decision_trace.score
        src = s.last_decision_trace.source
        board.place(move[0], move[1], current)
        label = f"  ply {ply}: {current.name} {move}  score={score}  src={src}"
        detail = eval_detail(board)
        print(f"{label}")
        print(f"         eval: {detail}")
        if board.check_win(move[0], move[1]):
            print(f"         *** {current.name} WINS ***")
            break
        current = Player.WHITE if current == Player.BLACK else Player.BLACK
    # undo all
    while board.move_history and board.move_history[-1] != (4, 4, Player.WHITE):
        board.undo()
    board.undo()  # undo (4,4) itself

    print("\n=== PART 4: PV for (4,2) — both sides use internal minimax depth=5 ===\n")
    board.place(4, 2, Player.WHITE)
    print(f"  After W(4,2): {eval_detail(board)}")
    current = Player.BLACK
    for ply in range(1, 10):
        s = AISearcher(depth=5, ai_player=current)
        move = s.find_best_move(board)
        if move is None:
            print(f"  ply {ply}: {current.name} has no move")
            break
        score = s.last_decision_trace.score
        src = s.last_decision_trace.source
        board.place(move[0], move[1], current)
        label = f"  ply {ply}: {current.name} {move}  score={score}  src={src}"
        detail = eval_detail(board)
        print(f"{label}")
        print(f"         eval: {detail}")
        if board.check_win(move[0], move[1]):
            print(f"         *** {current.name} WINS ***")
            break
        current = Player.WHITE if current == Player.BLACK else Player.BLACK
    while board.move_history and board.move_history[-1] != (4, 2, Player.WHITE):
        board.undo()
    board.undo()


if __name__ == "__main__":
    main()
