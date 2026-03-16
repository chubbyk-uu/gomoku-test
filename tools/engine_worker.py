"""Subprocess worker that serves Gomoku engine moves for a specific repo checkout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_board(repo_root: Path, moves: list[list[int]]):
    sys.path.insert(0, str(repo_root / "src"))
    from gomoku.board import Board
    from gomoku.config import Player

    board = Board()
    for row, col, player in moves:
        board.place(row, col, Player(player))
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve best-move requests for one repo version.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--ai-player", type=str, required=True, choices=("BLACK", "WHITE"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root / "src"))

    from gomoku.ai.searcher import AISearcher
    from gomoku.config import Player

    ai_player = Player[args.ai_player]
    searcher = AISearcher(depth=args.depth, ai_player=ai_player, time_limit_s=None)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        request = json.loads(line)
        cmd = request.get("cmd")
        if cmd == "quit":
            break
        if cmd != "best_move":
            print(json.dumps({"error": f"unsupported command: {cmd}"}), flush=True)
            continue

        try:
            board = _build_board(repo_root, request["moves"])
            move = searcher.find_best_move(board)
            print(
                json.dumps(
                    {
                        "move": list(move) if move is not None else None,
                        "stats": searcher.last_search_stats.__dict__,
                    }
                ),
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - defensive worker error path
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)


if __name__ == "__main__":
    main()
