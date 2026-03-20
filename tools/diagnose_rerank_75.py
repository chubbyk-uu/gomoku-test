"""Deep-dive on (7,5) opening move-8: why rerank pushes (4,2) down.

Reproduces the (7,5) game, stops before move 8, then:
1. Runs find_best_move(depth=5) with full root_candidates trace
2. Shows all rerank scores
3. Probes (4,2) vs (4,4) reply candidates manually
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
from repo_paths import DEFAULT_OPPONENT_REPO, REPO_ROOT  # noqa: E402

REPO_A = str(REPO_ROOT)
REPO_B = str(DEFAULT_OPPONENT_REPO)
DEPTH = 5
OPENING = (7, 5)

# Known move sequence for the (7,5) game up to move 7 (verified from previous run)
# Move 1: BLACK (7,5) fixed
# Move 2: WHITE (5,3)  minimax+rerank
# Move 3: BLACK (6,6)  minimax
# Move 4: WHITE (6,2)  minimax+rerank
# Move 5: BLACK (8,4)  minimax
# Move 6: WHITE (9,3)  vcf_block
# Move 7: BLACK (6,5)  minimax
# Move 8: WHITE ??? -> we want to reproduce this


def board_repr(board: Board) -> str:
    size = board.grid.shape[0]
    rows = ["   " + " ".join(f"{c:2d}" for c in range(size))]
    for r in range(size):
        cells = []
        for c in range(size):
            v = int(board.grid[r, c])
            cells.append(" B" if v == 1 else " W" if v == 2 else " .")
        rows.append(f"{r:2d} " + "".join(cells))
    return "\n".join(rows)


def replay_to_move7() -> Board:
    """Play the (7,5) opening game until just before WHITE's move 8."""
    sb = _EngineWrapper(DEPTH, Player.BLACK, REPO_B)
    sw = _EngineWrapper(DEPTH, Player.WHITE, REPO_A)

    board = Board()
    board.place(OPENING[0], OPENING[1], Player.BLACK)
    print(f"Move 1: BLACK {OPENING} [fixed]")

    current = Player.WHITE
    move_no = 1
    moves_played = []

    try:
        while move_no < 7:
            engine = sw if current == Player.WHITE else sb
            move = engine.find_best_move(board)
            if move is None:
                print("ERROR: No move found")
                break

            row, col = move
            move_no += 1
            board.place(row, col, current)
            moves_played.append((row, col, current))

            trace = engine.last_decision_trace
            src = trace.source if hasattr(trace, "source") else trace.get("source", "")
            print(f"Move {move_no}: {current.name} ({row},{col})  source={src}")

            current = Player.WHITE if current == Player.BLACK else Player.BLACK
    finally:
        sb.close()
        sw.close()

    print(f"\nBoard at move 8 (before WHITE plays):")
    print(board_repr(board))
    print(f"Move history: {[(r,c,p.name) for r,c,p in board.move_history]}")
    return board


def analyse_move8(board: Board) -> None:
    """Run find_best_move on the board and analyse rerank in detail."""
    from gomoku.ai.searcher import AISearcher, _EARLY_ROOT_RERANK_LAMBDA_MAX, _EARLY_ROOT_RERANK_LAMBDA_AVG

    searcher = AISearcher(depth=DEPTH, ai_player=Player.WHITE)
    chosen = searcher.find_best_move(board)
    trace = searcher.last_decision_trace

    print("\n" + "=" * 70)
    print(f"find_best_move -> {chosen}  source={trace.source}")
    print("=" * 70)

    candidates = trace.root_candidates or []
    print(f"\nFull root_candidates ({len(candidates)} total):")
    print(f"  {'#':>3}  {'move':>7}  {'score':>8}  {'base':>8}  {'max_rpl':>9}  {'avg_rpl':>9}  {'rerank':>10}  {'b_rank':>6}")
    print("  " + "-" * 74)
    for i, c in enumerate(candidates, 1):
        mv = c.get("move")
        score = c.get("score", "?")
        base = c.get("base_score", "—")
        mr = c.get("max_reply_score", "—")
        ar = c.get("avg_reply_score", "—")
        rs = c.get("rerank_score", "—")
        br = c.get("base_rank", "—")
        chosen_mark = " *** CHOSEN" if mv and tuple(mv) == chosen else ""
        mark42 = " <-- (4,2)" if mv and mv[0] == 4 and mv[1] == 2 else ""
        mark44 = " <-- (4,4)" if mv and mv[0] == 4 and mv[1] == 4 else ""
        print(f"  {i:>3}  {str(mv):>7}  {str(score):>8}  {str(base):>8}  {str(mr):>9}  {str(ar):>9}  {str(rs):>10}  {str(br):>6}{chosen_mark}{mark42}{mark44}")

    # Deep dive on (4,2) and (4,4)
    c42 = next((c for c in candidates if c.get("move") == [4, 2]), None)
    c44 = next((c for c in candidates if c.get("move") == [4, 4]), None)

    print(f"\n{'='*70}")
    print("(4,2) detail:")
    if c42:
        print(json.dumps(c42, indent=2))
    else:
        print("  NOT FOUND in root_candidates")

    print(f"\n(4,4) detail:")
    if c44:
        print(json.dumps(c44, indent=2))
    else:
        print("  NOT FOUND in root_candidates")

    # Now manually probe to confirm which is actually better
    print(f"\n{'='*70}")
    print("Manual verification: force (4,2) vs (4,4) and play out vs zhou:")
    for target in [(4, 2), (4, 4)]:
        row, col = target
        if board.grid[row, col] != 0:
            print(f"  {target}: OCCUPIED, skip")
            continue
        board.place(row, col, Player.WHITE)
        print(f"\n  After WHITE plays {target}:")
        _play_short(board.copy(), target)
        board.undo()


def _play_short(board: Board, first_move: tuple[int, int]) -> None:
    """Play 6 more moves (3 each) from this position and show result."""
    sb = _EngineWrapper(DEPTH, Player.BLACK, REPO_B)
    sw = _EngineWrapper(DEPTH, Player.WHITE, REPO_A)
    current = Player.BLACK
    moves = [first_move]

    try:
        for _ in range(12):  # up to 12 more moves
            engine = sb if current == Player.BLACK else sw
            move = engine.find_best_move(board)
            if move is None:
                print(f"    -> No move for {current.name}")
                break
            row, col = move
            board.place(row, col, current)
            moves.append((row, col))
            src = engine.last_decision_trace
            src_str = (src.source if hasattr(src, "source") else src.get("source", "?")) if src else "?"
            print(f"    {current.name} ({row},{col}) src={src_str}")
            if board.check_win(row, col):
                print(f"    -> {current.name} WINS!")
                break
            current = Player.WHITE if current == Player.BLACK else Player.BLACK
    finally:
        sb.close()
        sw.close()


def main() -> None:
    board = replay_to_move7()
    analyse_move8(board)


if __name__ == "__main__":
    main()
