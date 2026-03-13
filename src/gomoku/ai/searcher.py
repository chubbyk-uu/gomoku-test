"""Minimax searcher with alpha-beta pruning and transposition table for Gomoku AI."""

import math
from typing import Optional

from gomoku.ai.evaluator import evaluate
from gomoku.board import Board
from gomoku.config import AI_MAX_CANDIDATES, Player

# 置换表条目：(depth, score, flag, best_move)
# flag: "E"=exact, "L"=lower bound (V >= score), "U"=upper bound (V <= score)
_TTEntry = tuple[int, float, str, Optional[tuple[int, int]]]


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

        每次调用前清空置换表，避免跨局面的哈希污染。

        Args:
            board: 当前棋盘状态（不会被修改）。

        Returns:
            最优落子坐标 (row, col)；无候选点时返回 None。
        """
        self._tt: dict[int, _TTEntry] = {}
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
        """Minimax 递归搜索（Alpha-Beta 剪枝 + 置换表）。

        置换表 flag 语义：
          "E" (exact)       —— V = score，精确值，可直接返回。
          "L" (lower bound) —— V >= score，maximizing 节点发生 beta 截断。
          "U" (upper bound) —— V <= score，maximizing 无截断但未能超过 alpha_orig；
                              或 minimizing 节点发生 alpha 截断。

        Args:
            board: 当前棋盘（使用 place/undo 原地修改，搜索结束后复原）。
            depth: 剩余搜索深度。
            alpha: Alpha 值（maximizer 的下界）。
            beta: Beta 值（minimizer 的上界）。
            maximizing: True 表示当前是 AI（最大化）回合。

        Returns:
            (评分, 最优落子) 的二元组；叶节点时落子为 None。
        """
        h = board.hash
        alpha_orig = alpha  # 保存调用方传入的 alpha，供事后判断 flag

        # --- 置换表查询 ---
        entry = self._tt.get(h)
        if entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = entry
            if tt_depth >= depth:
                if tt_flag == "E":
                    return tt_score, tt_move
                if tt_flag == "L":
                    alpha = max(alpha, tt_score)
                else:  # "U"
                    beta = min(beta, tt_score)
                if beta <= alpha:
                    return tt_score, tt_move

        # --- 叶节点 ---
        if depth == 0:
            score = evaluate(board, self.ai_player)
            self._tt[h] = (0, score, "E", None)
            return score, None

        # --- 无候选点 ---
        moves = board.get_candidate_moves()
        if not moves:
            score = evaluate(board, self.ai_player)
            self._tt[h] = (depth, score, "E", None)
            return score, None

        # --- 候选点排序：模拟落子后快速评估，按对当前方有利程度降序排列；截断到前 N 个 ---
        current_player = self.ai_player if maximizing else self._opponent
        scored_moves: list[tuple[int, int, int]] = []
        for r, c in moves:
            board.place(r, c, current_player)
            score = evaluate(board, self.ai_player)
            board.undo()
            scored_moves.append((r, c, score))
        # 坐标作为 tiebreaker 保证排序稳定（set 迭代顺序不确定）
        scored_moves.sort(key=lambda x: (x[2], x[0], x[1]), reverse=maximizing)
        moves = [(r, c) for r, c, _ in scored_moves[:AI_MAX_CANDIDATES]]

        best_move: Optional[tuple[int, int]] = None

        if maximizing:
            best_score: float = -math.inf
            for row, col in moves:
                board.place(row, col, current_player)
                if board.check_win(row, col):
                    board.undo()
                    self._tt[h] = (depth, 100_000.0, "E", (row, col))
                    return 100_000.0, (row, col)
                score, _ = self._minimax(board, depth - 1, alpha, beta, False)
                board.undo()
                if score > best_score:
                    best_score = score
                    best_move = (row, col)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    # beta 截断：真实值 >= best_score，存为下界
                    self._tt[h] = (depth, best_score, "L", best_move)
                    return best_score, best_move
            # 无截断：判断是精确值还是上界（未能超过 alpha_orig）
            flag = "U" if best_score <= alpha_orig else "E"
            self._tt[h] = (depth, best_score, flag, best_move)
            return best_score, best_move
        else:
            best_score = math.inf
            for row, col in moves:
                board.place(row, col, current_player)
                if board.check_win(row, col):
                    board.undo()
                    self._tt[h] = (depth, -100_000.0, "E", (row, col))
                    return -100_000.0, (row, col)
                score, _ = self._minimax(board, depth - 1, alpha, beta, True)
                board.undo()
                if score < best_score:
                    best_score = score
                    best_move = (row, col)
                beta = min(beta, best_score)
                if beta <= alpha:
                    # alpha 截断：真实值 <= best_score，存为上界
                    self._tt[h] = (depth, best_score, "U", best_move)
                    return best_score, best_move
            # 无截断：所有候选点均已遍历，best_score 是精确最小值
            self._tt[h] = (depth, best_score, "E", best_move)
            return best_score, best_move
