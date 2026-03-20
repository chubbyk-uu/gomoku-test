"""Find current failing games in the left-column openings (col=5 of center 5x5 matrix).

Runs BLACK at col=5 openings: (5,5),(6,5),(7,5),(8,5),(9,5)
WHITE=gomoku-test(depth=5), BLACK=zhou(depth=5)

For each game:
- Records result and per-move trace.source
- Prints first divergent move if same prefix as winning game
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark import _EngineWrapper  # noqa: E402
from gomoku.board import Board  # noqa: E402
from gomoku.config import Player  # noqa: E402
from repo_paths import DEFAULT_OPPONENT_REPO, REPO_ROOT  # noqa: E402

REPO_A = str(REPO_ROOT)  # gomoku-test = WHITE
REPO_B = str(DEFAULT_OPPONENT_REPO)  # zhou = BLACK
DEPTH = 5
MAX_MOVES = 80

LEFT_COL_OPENINGS = [(5, 5), (6, 5), (7, 5), (8, 5), (9, 5)]


def play_fixed(opening: tuple[int, int]) -> dict:
    sb = _EngineWrapper(DEPTH, Player.BLACK, REPO_B)
    sw = _EngineWrapper(DEPTH, Player.WHITE, REPO_A)
    board = Board()
    board.place(opening[0], opening[1], Player.BLACK)

    moves = [{"move_no": 1, "player": "BLACK", "coord": opening, "source": "fixed"}]
    current = Player.WHITE
    move_no = 1
    winner = None

    try:
        while move_no < MAX_MOVES:
            engine = sw if current == Player.WHITE else sb
            move = engine.find_best_move(board)
            if move is None:
                break

            row, col = move
            move_no += 1
            board.place(row, col, current)

            trace = engine.last_decision_trace
            src = ""
            if isinstance(trace, dict):
                src = trace.get("source", "")
            elif hasattr(trace, "source"):
                src = trace.source

            moves.append({
                "move_no": move_no,
                "player": current.name,
                "coord": (row, col),
                "source": src,
                "root_candidates": (
                    (trace.root_candidates if hasattr(trace, "root_candidates") else trace.get("root_candidates"))
                    if trace else None
                ),
                "score": (
                    (trace.score if hasattr(trace, "score") else trace.get("score"))
                    if trace else None
                ),
            })

            if board.check_win(row, col):
                winner = current.name
                break

            current = Player.WHITE if current == Player.BLACK else Player.BLACK
    finally:
        sb.close()
        sw.close()

    return {"opening": opening, "winner": winner, "moves": moves}


def print_board(board: Board) -> None:
    size = board.grid.shape[0]
    header = "   " + " ".join(f"{c:2d}" for c in range(size))
    print(header)
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
        print(f"{r:2d} " + "".join(cells))


def main() -> None:
    results = []
    for opening in LEFT_COL_OPENINGS:
        t0 = time.perf_counter()
        result = play_fixed(opening)
        elapsed = time.perf_counter() - t0
        results.append(result)
        winner = result["winner"] or "DRAW"
        n = len(result["moves"])
        print(f"Opening {opening}: winner={winner}  moves={n}  ({elapsed:.1f}s)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    wins = sum(1 for r in results if r["winner"] == "WHITE")
    losses = sum(1 for r in results if r["winner"] == "BLACK")
    draws = sum(1 for r in results if r["winner"] is None)
    print(f"WHITE (gomoku-test): {wins}W {losses}L {draws}D")

    # For losing games, show detailed per-move trace sources
    print("\n--- Failing games detail ---")
    for r in results:
        if r["winner"] != "BLACK":
            continue
        print(f"\nOpening {r['opening']} — WHITE LOST in {len(r['moves'])} moves")
        for m in r["moves"]:
            src = m["source"]
            mv = m["coord"]
            player = m["player"]
            mark = ""
            if player == "WHITE" and src == "minimax":
                cands = m.get("root_candidates") or []
                # check if rerank was applied
                has_rerank = any("rerank_score" in c for c in cands) if cands else False
                mark = " [MINIMAX" + ("+RERANK" if has_rerank else "") + "]"
            print(f"  Move {m['move_no']:2d}: {player:5s} {str(mv):8s}  source={src}{mark}")

    # For the first failing game, replay to the first WHITE minimax decision and show candidates
    print("\n--- First failing game: first WHITE minimax move candidates ---")
    failing = [r for r in results if r["winner"] == "BLACK"]
    if not failing:
        print("No failing games found! All won.")
        return

    first_fail = failing[0]
    print(f"Opening {first_fail['opening']}")
    for m in first_fail["moves"]:
        if m["player"] == "WHITE" and m["source"] == "minimax":
            cands = m.get("root_candidates") or []
            print(f"\n  Move {m['move_no']} WHITE minimax -> {m['coord']}")
            if not cands:
                print("  (no root_candidates in trace)")
                continue
            has_rerank = any("rerank_score" in c for c in cands)
            print(f"  has_rerank={has_rerank}  total_candidates={len(cands)}")
            print(f"  {'rank':>4}  {'move':>8}  {'score':>10}  {'base_score':>10}  {'rerank_score':>13}  {'max_reply':>10}  {'base_rank':>9}")
            print("  " + "-" * 72)
            for i, c in enumerate(cands[:10], 1):
                mv = c.get("move", [None, None])
                score = c.get("score", "?")
                base = c.get("base_score", "—")
                rs = c.get("rerank_score", "—")
                mr = c.get("max_reply_score", "—")
                br = c.get("base_rank", "—")
                print(f"  {i:>4}  {str(mv):>8}  {str(score):>10}  {str(base):>10}  {str(rs):>13}  {str(mr):>10}  {str(br):>9}")
            break


if __name__ == "__main__":
    main()
