"""Diagnose the move-8 rerank failure for the (4,4) opening (A=WHITE vs zhou=BLACK).

This script:
1. Plays the (4,4) fixed opening game up to and including move 7
2. At move 8, runs find_best_move(depth=5) on the WHITE engine (gomoku-test)
3. Prints the full root_candidates list with rerank scores
4. Highlights why (4,2) gets pushed down
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark import _EngineWrapper  # noqa: E402

from gomoku.board import Board  # noqa: E402
from gomoku.config import Player  # noqa: E402

REPO_A = str(Path(__file__).parent.parent.resolve())   # gomoku-test
REPO_B = str(Path("/home/jerry/python-test/gomoku/zhou").resolve())  # zhou

OPENING = (4, 4)
DEPTH = 5
MAX_MOVES = 9   # 只玩到 move 8 就停下分析


def _board_repr(board: Board) -> str:
    """Print board with move numbers."""
    size = board.grid.shape[0]
    rows = []
    header = "   " + " ".join(f"{c:2d}" for c in range(size))
    rows.append(header)
    for r in range(size):
        cells = []
        for c in range(size):
            v = int(board.grid[r, c])
            if v == int(Player.BLACK):
                cells.append(" B")
            elif v == int(Player.WHITE):
                cells.append(" W")
            else:
                cells.append(" .")
        rows.append(f"{r:2d} " + "".join(cells))
    return "\n".join(rows)


def main() -> None:
    sb = _EngineWrapper(DEPTH, Player.BLACK, REPO_B)   # zhou plays BLACK
    sw = _EngineWrapper(DEPTH, Player.WHITE, REPO_A)   # gomoku-test plays WHITE

    board = Board()
    board.place(OPENING[0], OPENING[1], Player.BLACK)
    print(f"Move 1: BLACK (zhou) places at {OPENING} [fixed opening]")
    move_history = [(OPENING[0], OPENING[1], Player.BLACK)]

    current = Player.WHITE
    move_no = 1

    try:
        while move_no < MAX_MOVES:
            engine = sw if current == Player.WHITE else sb
            label = "gomoku-test(W)" if current == Player.WHITE else "zhou(B)"

            move = engine.find_best_move(board)
            if move is None:
                print("No move found — game over")
                break

            row, col = move
            move_no += 1
            board.place(row, col, current)
            move_history.append((row, col, current))

            trace = engine.last_decision_trace
            trace_src = ""
            if isinstance(trace, dict):
                trace_src = trace.get("source", "")
            elif hasattr(trace, "source"):
                trace_src = trace.source

            print(f"Move {move_no}: {label} plays ({row},{col})  source={trace_src}")

            if board.check_win(row, col):
                print(f"  -> {label} wins!")
                break

            if move_no == 8:
                # This IS the move — we just played it; now analyse what happened
                _analyse_move8(board, current, move, engine)
                break

            current = Player.WHITE if current == Player.BLACK else Player.BLACK

    finally:
        sb.close()
        sw.close()


def _analyse_move8(board: Board, player: Player, actual_move: tuple[int, int], engine) -> None:
    """Analyse the move-8 decision: show root_candidates with rerank scores."""
    print("\n" + "=" * 70)
    print(f"MOVE 8 ANALYSIS — player={player.name}  actual_move={actual_move}")
    print("=" * 70)

    # Board BEFORE this move (undo it temporarily just to print the pre-move board)
    board.undo()
    print("\nBoard at move-8 position (before WHITE plays):")
    print(_board_repr(board))

    # Now re-run find_best_move to capture the trace freshly
    from gomoku.ai.searcher import AISearcher
    searcher = AISearcher(depth=5, ai_player=Player.WHITE)
    chosen = searcher.find_best_move(board)
    trace = searcher.last_decision_trace

    print(f"\nfind_best_move chose: {chosen}")
    print(f"source: {trace.source}")
    print(f"completed_depth: {trace.completed_depth}")
    print(f"score: {trace.score}")

    candidates = trace.root_candidates or []
    print(f"\nroot_candidates ({len(candidates)} total):")
    print(f"{'rank':>5}  {'move':>10}  {'base_score':>12}  {'max_reply':>12}  {'avg_reply':>12}  {'rerank_score':>13}  {'base_rank':>10}")
    print("-" * 90)

    for i, c in enumerate(candidates, 1):
        move = c.get("move", [None, None])
        base_score = c.get("base_score", c.get("score", "?"))
        max_reply = c.get("max_reply_score", "—")
        avg_reply = c.get("avg_reply_score", "—")
        rerank_score = c.get("rerank_score", "—")
        base_rank = c.get("base_rank", "—")
        tag = " *** CHOSEN ***" if move and tuple(move) == chosen else ""
        tag42 = " <-- (4,2)" if move and move[0] == 4 and move[1] == 2 else ""
        print(f"{i:>5}  {str(move):>10}  {str(base_score):>12}  {str(max_reply):>12}  {str(avg_reply):>12}  {str(rerank_score):>13}  {str(base_rank):>10}{tag}{tag42}")

    # Find (4,2) specifically
    print("\n--- (4,2) detail ---")
    for c in candidates:
        mv = c.get("move", [])
        if isinstance(mv, list) and len(mv) == 2 and mv[0] == 4 and mv[1] == 2:
            print(json.dumps(c, indent=2))
            break
    else:
        print("(4,2) NOT found in root_candidates")
        print("Candidate moves present:", [c.get("move") for c in candidates])

    # Also show notes (lambda values)
    if trace.notes:
        print("\nnotes:", trace.notes)


if __name__ == "__main__":
    main()
