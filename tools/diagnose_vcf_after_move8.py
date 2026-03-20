"""After white plays each move-8 candidate, check if black has a VCF forced win.

Tests candidates at multiple VCF depths to understand sensitivity.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gomoku.ai.vcf import VCFSolver
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

CANDIDATES = [
    (7, 1),  # minimax #1 — known loser
    (4, 2),  # minimax #2 — known winner
    (6, 3),  # minimax #3 — known loser
    (4, 4),  # minimax #4 — known loser
    (9, 5),  # minimax #5 — known loser
]

VCF_DEPTHS = [4, 6, 8, 10, 12]


def build_board() -> Board:
    board = Board()
    for row, col, player in SEQUENCE_TO_MOVE7:
        board.place(row, col, player)
    return board


def check_black_vcf(board: Board, white_move: tuple[int, int], vcf_depth: int) -> tuple[bool, object]:
    """Place white_move, then check if black has VCF win. Returns (found, winning_move)."""
    board.place(white_move[0], white_move[1], Player.WHITE)
    solver = VCFSolver()
    win = solver.find_winning_move(board, Player.BLACK, vcf_depth)
    board.undo()
    return (win is not None), win


def main() -> None:
    board = build_board()

    print(f"Move-8 candidates: black VCF check after each white move\n")
    print(f"{'move':>7} | " + " | ".join(f"vcf_d={d}" for d in VCF_DEPTHS))
    print("-" * 70)

    for candidate in CANDIDATES:
        results = []
        for depth in VCF_DEPTHS:
            found, win_move = check_black_vcf(board, candidate, depth)
            results.append(f"{'YES ' + str(win_move):>16}" if found else f"{'no':>16}")
        print(f"{str(candidate):>7} | " + " | ".join(results))

    # Also show detailed trace for (4,4) and (4,2) at max depth
    print(f"\n=== Detailed VCF trace: BLACK after WHITE plays (4,4) ===")
    board.place(4, 4, Player.WHITE)
    solver = VCFSolver()
    win = solver.find_winning_move(board, Player.BLACK, 12)
    stats = solver.last_stats
    print(f"  Result: {win}")
    print(f"  nodes={stats.nodes}  cache_hits={stats.cache_hits}  "
          f"attack_candidates={stats.attack_candidates}  max_depth_reached={stats.max_depth_reached}")
    if win and hasattr(solver, 'last_trace') and solver.last_trace:
        tr = solver.last_trace
        attacks = getattr(tr, 'top_level_attacks', [])
        print(f"  top_level_attacks: {attacks[:5]}")
    board.undo()

    print(f"\n=== Detailed VCF trace: BLACK after WHITE plays (4,2) ===")
    board.place(4, 2, Player.WHITE)
    solver2 = VCFSolver()
    win2 = solver2.find_winning_move(board, Player.BLACK, 12)
    stats2 = solver2.last_stats
    print(f"  Result: {win2}")
    print(f"  nodes={stats2.nodes}  cache_hits={stats2.cache_hits}  "
          f"attack_candidates={stats2.attack_candidates}  max_depth_reached={stats2.max_depth_reached}")
    board.undo()


if __name__ == "__main__":
    main()
