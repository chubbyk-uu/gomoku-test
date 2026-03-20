"""Test all top move-8 white candidates against zhou (full game).

Usage:
    python tools/test_move8_candidates.py [move_row,move_col ...]

If no arguments, tests all top candidates from diagnose output.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark import _EngineWrapper
from gomoku.board import Board
from gomoku.config import Player
from repo_paths import DEFAULT_OPPONENT_REPO, REPO_ROOT

REPO_A = str(REPO_ROOT)
REPO_B = str(DEFAULT_OPPONENT_REPO)
DEPTH = 5
OPENING = (7, 5)
MAX_MOVES = 80

# All top candidates to test (by base minimax rank)
DEFAULT_CANDIDATES = [
    (7, 1),  # base #1, score=740
    (4, 2),  # base #2, score=568
    (6, 3),  # base #3, score=538
    (4, 4),  # base #4, score=411  (known loser, as baseline)
    (9, 5),  # base #5, score=324
    (5, 5),  # base #6, score=235
    (5, 4),  # base #7, score=191
]


def replay_to_move7() -> Board:
    """Replay (7,5) opening to just before WHITE's move 8."""
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


def play_full_game(board: Board, white_first_move: tuple[int, int]) -> str:
    """Play a full game from board state, white plays white_first_move first.

    Returns: 'WHITE', 'BLACK', or 'DRAW'
    """
    sb = _EngineWrapper(DEPTH, Player.BLACK, REPO_B)
    sw = _EngineWrapper(DEPTH, Player.WHITE, REPO_A)

    row, col = white_first_move
    board.place(row, col, Player.WHITE)
    moves = [f"W{white_first_move}"]

    if board.check_win(row, col):
        sb.close()
        sw.close()
        return "WHITE"

    current = Player.BLACK
    try:
        while len(moves) < MAX_MOVES:
            engine = sb if current == Player.BLACK else sw
            move = engine.find_best_move(board)
            if move is None:
                return "DRAW"
            r, c = move
            board.place(r, c, current)
            moves.append(f"{'B' if current == Player.BLACK else 'W'}{(r,c)}")
            if board.check_win(r, c):
                return current.name
            current = Player.WHITE if current == Player.BLACK else Player.BLACK
    finally:
        sb.close()
        sw.close()

    return "DRAW"


def test_candidate(board_snapshot: Board, move: tuple[int, int]) -> None:
    board = board_snapshot.copy()
    result = play_full_game(board, move)
    symbol = "WIN " if result == "WHITE" else ("LOSS" if result == "BLACK" else "DRAW")
    print(f"  {symbol}  WHITE plays {move} -> {result}", flush=True)


def main() -> None:
    if len(sys.argv) > 1:
        candidates = []
        for arg in sys.argv[1:]:
            r, c = arg.split(",")
            candidates.append((int(r), int(c)))
    else:
        candidates = DEFAULT_CANDIDATES

    print("Replaying to move 7...", flush=True)
    board = replay_to_move7()
    print(f"Board ready. Testing {len(candidates)} candidates vs zhou (depth={DEPTH}):\n")

    for move in candidates:
        test_candidate(board, move)


if __name__ == "__main__":
    main()
