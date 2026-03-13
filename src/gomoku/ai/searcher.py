"""Minimax searcher with alpha-beta pruning and transposition table for Gomoku AI."""

import math
import time
from dataclasses import dataclass
from typing import Optional

from gomoku.ai.evaluator import evaluate
from gomoku.board import Board
from gomoku.config import AI_EVAL_CACHE_MAX_SIZE, AI_MAX_CANDIDATES, AI_TT_MAX_SIZE, Player

# 置换表条目：(depth, score, flag, best_move)
# flag: "E"=exact, "L"=lower bound (V >= score), "U"=upper bound (V <= score)
_TTEntry = tuple[int, float, str, Optional[tuple[int, int]]]


@dataclass
class SearchStats:
    """一次搜索的统计信息，用于 profiling 和回归观测。"""

    nodes: int = 0
    leaf_evals: int = 0
    ordering_evals: int = 0
    tt_hits: int = 0
    tt_cutoffs: int = 0
    beta_cutoffs: int = 0
    alpha_cutoffs: int = 0
    immediate_wins: int = 0
    max_branching: int = 0
    completed_depth: int = 0
    timed_out: bool = False

    def merge(self, other: "SearchStats") -> None:
        """累加一次迭代的统计到总统计中。"""
        self.nodes += other.nodes
        self.leaf_evals += other.leaf_evals
        self.ordering_evals += other.ordering_evals
        self.tt_hits += other.tt_hits
        self.tt_cutoffs += other.tt_cutoffs
        self.beta_cutoffs += other.beta_cutoffs
        self.alpha_cutoffs += other.alpha_cutoffs
        self.immediate_wins += other.immediate_wins
        self.max_branching = max(self.max_branching, other.max_branching)
        self.completed_depth = max(self.completed_depth, other.completed_depth)
        self.timed_out = self.timed_out or other.timed_out


class SearchTimeout(Exception):
    """Raised when iterative deepening exceeds the configured time budget."""


class AISearcher:
    """基于 Minimax + Alpha-Beta 剪枝的五子棋 AI 搜索器。

    Attributes:
        depth: 搜索深度（建议 2~4，>3 时速度明显下降）。
        ai_player: AI 执棋颜色。
    """

    def __init__(
        self,
        depth: int = 3,
        ai_player: Player = Player.WHITE,
        time_limit_s: Optional[float] = None,
    ) -> None:
        self.depth = depth
        self.ai_player = ai_player
        self.time_limit_s = time_limit_s
        self._opponent = Player.WHITE if ai_player == Player.BLACK else Player.BLACK
        self._tt: dict[int, _TTEntry] = {}
        self._eval_cache: dict[int, int] = {}
        self.last_search_stats = SearchStats()
        self._deadline: Optional[float] = None

    def find_best_move(self, board: Board) -> Optional[tuple[int, int]]:
        """为 AI 找出当前局面下的最优落子位置。

        置换表和评估缓存会在同一局内复用，避免重复搜索/重复评估。
        它们都以局面哈希为 key，不会改变搜索结果，只减少重复工作。

        Args:
            board: 当前棋盘状态（不会被修改）。

        Returns:
            最优落子坐标 (row, col)；无候选点时返回 None。
        """
        self.last_search_stats = SearchStats()
        self._deadline = (
            time.perf_counter() + self.time_limit_s if self.time_limit_s is not None else None
        )

        best_move: Optional[tuple[int, int]] = None
        for current_depth in range(1, self.depth + 1):
            iteration_stats = SearchStats()
            try:
                score, move = self._minimax(
                    board,
                    current_depth,
                    -math.inf,
                    math.inf,
                    maximizing=True,
                    stats=iteration_stats,
                )
            except SearchTimeout:
                self.last_search_stats.timed_out = True
                break

            iteration_stats.completed_depth = current_depth
            self.last_search_stats.merge(iteration_stats)
            best_move = move

            if abs(score) >= 100_000.0:
                break

            if self._deadline is not None and time.perf_counter() >= self._deadline:
                self.last_search_stats.timed_out = True
                break

        self._deadline = None
        return best_move

    def _evaluate(self, board: Board) -> int:
        """返回当前局面的评估值，并缓存结果以复用。"""
        h = board.hash
        cached = self._eval_cache.get(h)
        if cached is not None:
            return cached
        score = evaluate(board, self.ai_player)
        if len(self._eval_cache) >= AI_EVAL_CACHE_MAX_SIZE:
            self._eval_cache.clear()
        self._eval_cache[h] = score
        return score

    @staticmethod
    def _store_tt(
        tt: dict[int, _TTEntry],
        h: int,
        entry: _TTEntry,
    ) -> None:
        """写入置换表；超限时先清空，避免长局无限增长。"""
        if len(tt) >= AI_TT_MAX_SIZE:
            tt.clear()
        tt[h] = entry

    def _check_timeout(self) -> None:
        """Raise when the configured time budget has been exhausted."""
        if self._deadline is not None and time.perf_counter() >= self._deadline:
            raise SearchTimeout

    def _find_immediate_winning_move(
        self,
        board: Board,
        moves: list[tuple[int, int]],
        current_player: Player,
    ) -> Optional[tuple[int, int]]:
        """检查候选点中是否存在一步直接成五的着法。"""
        for row, col in moves:
            board.place(row, col, current_player)
            try:
                is_win = board.check_win(row, col)
            finally:
                board.undo()
            if is_win:
                return row, col
        return None

    @staticmethod
    def _prioritize_tt_move(
        moves: list[tuple[int, int]],
        tt_move: Optional[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """若 TT 提供了候选中的 best move，则将其提前到首位。"""
        if tt_move is None:
            return moves
        try:
            idx = moves.index(tt_move)
        except ValueError:
            return moves
        if idx == 0:
            return moves
        return [moves[idx], *moves[:idx], *moves[idx + 1 :]]

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
        stats: SearchStats,
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
        self._check_timeout()
        stats.nodes += 1
        h = board.hash
        alpha_orig = alpha  # 保存调用方传入的 alpha，供事后判断 flag

        # --- 置换表查询 ---
        entry = self._tt.get(h)
        if entry is not None:
            stats.tt_hits += 1
            tt_depth, tt_score, tt_flag, tt_move = entry
            if tt_depth >= depth:
                if tt_flag == "E":
                    return tt_score, tt_move
                if tt_flag == "L":
                    alpha = max(alpha, tt_score)
                else:  # "U"
                    beta = min(beta, tt_score)
                if beta <= alpha:
                    stats.tt_cutoffs += 1
                    return tt_score, tt_move

        # --- 叶节点 ---
        if depth == 0:
            stats.leaf_evals += 1
            score = self._evaluate(board)
            self._store_tt(self._tt, h, (0, score, "E", None))
            return score, None

        # --- 无候选点 ---
        moves = board.get_candidate_moves()
        if not moves:
            stats.leaf_evals += 1
            score = self._evaluate(board)
            self._store_tt(self._tt, h, (depth, score, "E", None))
            return score, None

        current_player = self.ai_player if maximizing else self._opponent
        stats.max_branching = max(stats.max_branching, len(moves))
        tt_move = entry[3] if entry is not None else None

        # --- 候选点预检查：若当前走子方存在一步直接成五，则无需再做完整排序 ---
        winning_move = self._find_immediate_winning_move(board, moves, current_player)
        if winning_move is not None:
            stats.immediate_wins += 1
            score = 100_000.0 if maximizing else -100_000.0
            self._store_tt(self._tt, h, (depth, score, "E", winning_move))
            return score, winning_move

        # --- 候选点排序：模拟落子后快速评估，按对当前方有利程度降序排列；截断到前 N 个 ---
        scored_moves: list[tuple[int, int, int]] = []
        for r, c in moves:
            self._check_timeout()
            board.place(r, c, current_player)
            try:
                stats.ordering_evals += 1
                score = self._evaluate(board)
            finally:
                board.undo()
            scored_moves.append((r, c, score))
        # 坐标作为 tiebreaker 保证排序稳定（set 迭代顺序不确定）
        scored_moves.sort(key=lambda x: (x[2], x[0], x[1]), reverse=maximizing)
        moves = [(r, c) for r, c, _ in scored_moves[:AI_MAX_CANDIDATES]]
        moves = self._prioritize_tt_move(moves, tt_move)

        best_move: Optional[tuple[int, int]] = None

        if maximizing:
            best_score: float = -math.inf
            for row, col in moves:
                self._check_timeout()
                board.place(row, col, current_player)
                try:
                    score, _ = self._minimax(board, depth - 1, alpha, beta, False, stats)
                finally:
                    board.undo()
                if score > best_score:
                    best_score = score
                    best_move = (row, col)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    # beta 截断：真实值 >= best_score，存为下界
                    stats.beta_cutoffs += 1
                    self._store_tt(self._tt, h, (depth, best_score, "L", best_move))
                    return best_score, best_move
            # 无截断：判断是精确值还是上界（未能超过 alpha_orig）
            flag = "U" if best_score <= alpha_orig else "E"
            self._store_tt(self._tt, h, (depth, best_score, flag, best_move))
            return best_score, best_move
        else:
            best_score = math.inf
            for row, col in moves:
                self._check_timeout()
                board.place(row, col, current_player)
                try:
                    score, _ = self._minimax(board, depth - 1, alpha, beta, True, stats)
                finally:
                    board.undo()
                if score < best_score:
                    best_score = score
                    best_move = (row, col)
                beta = min(beta, best_score)
                if beta <= alpha:
                    # alpha 截断：真实值 <= best_score，存为上界
                    stats.alpha_cutoffs += 1
                    self._store_tt(self._tt, h, (depth, best_score, "U", best_move))
                    return best_score, best_move
            # 无截断：所有候选点均已遍历，best_score 是精确最小值
            self._store_tt(self._tt, h, (depth, best_score, "E", best_move))
            return best_score, best_move
