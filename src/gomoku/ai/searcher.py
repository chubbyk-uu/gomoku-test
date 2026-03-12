"""Minimax searcher with alpha-beta pruning for Gomoku AI."""

import math
from typing import Optional

from gomoku.ai.evaluator import evaluate
from gomoku.board import Board
from gomoku.config import Player


class AISearcher:
    """基于 Minimax + Alpha-Beta 剪枝的五子棋 AI 搜索器。

    Attributes:
        depth: 搜索深度（建议 2~4，>3 时速度明显下降）。
        ai_player: AI 执棋颜色。
    """

    def __init__(self, depth: int = 3, ai_player: Player = Player.WHITE) -> None:
        self.depth = depth
        self.ai_player = ai_player
        self._opponent = Player.WHITE if ai_player == Player.BLACK else Player.BLACK

    def find_best_move(self, board: Board) -> Optional[tuple[int, int]]:
        """为 AI 找出当前局面下的最优落子位置。

        Args:
            board: 当前棋盘状态（不会被修改）。

        Returns:
            最优落子坐标 (row, col)；无候选点时返回 None。
        """
        _, move = self._minimax(board, self.depth, -math.inf, math.inf, maximizing=True)
        return move

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _minimax(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
    ) -> tuple[float, Optional[tuple[int, int]]]:
        """Minimax 递归搜索（Alpha-Beta 剪枝）。

        Args:
            board: 当前棋盘（使用 place/undo 原地修改，搜索结束后复原）。
            depth: 剩余搜索深度。
            alpha: Alpha 值（maximizer 的下界）。
            beta: Beta 值（minimizer 的上界）。
            maximizing: True 表示当前是 AI（最大化）回合。

        Returns:
            (评分, 最优落子) 的二元组；叶节点时落子为 None。
        """
        if depth == 0:
            return evaluate(board, self.ai_player), None

        moves = board.get_candidate_moves()
        if not moves:
            return evaluate(board, self.ai_player), None

        # 候选点排序：模拟落子后快速评估，按对当前方有利程度降序排列
        current_player = self.ai_player if maximizing else self._opponent
        scored_moves: list[tuple[int, int, int]] = []
        for r, c in moves:
            board.place(r, c, current_player)
            score = evaluate(board, self.ai_player)
            board.undo()
            scored_moves.append((r, c, score))
        # maximizing 方希望分高的先搜，minimizing 方希望分低的先搜
        scored_moves.sort(key=lambda x: x[2], reverse=maximizing)
        moves = [(r, c) for r, c, _ in scored_moves]

        best_move: Optional[tuple[int, int]] = None

        if maximizing:
            best_score: float = -math.inf
            for row, col in moves:
                board.place(row, col, current_player)
                if board.check_win(row, col):
                    board.undo()
                    return 100_000.0, (row, col)
                score, _ = self._minimax(board, depth - 1, alpha, beta, False)
                board.undo()
                if score > best_score:
                    best_score = score
                    best_move = (row, col)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return best_score, best_move
        else:
            best_score = math.inf
            for row, col in moves:
                board.place(row, col, current_player)
                if board.check_win(row, col):
                    board.undo()
                    return -100_000.0, (row, col)
                score, _ = self._minimax(board, depth - 1, alpha, beta, True)
                board.undo()
                if score < best_score:
                    best_score = score
                    best_move = (row, col)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return best_score, best_move
