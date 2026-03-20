"""Minimax searcher with alpha-beta pruning and transposition table for Gomoku AI."""

import math
import time
from dataclasses import dataclass
from typing import Optional

from gomoku.ai.evaluator import DEFENSE_WEIGHT, evaluate
from gomoku.ai.vcf import VCFSolver
from gomoku.board import Board
from gomoku.config import (
    AI_EVAL_CACHE_MAX_SIZE,
    AI_MAX_CANDIDATES,
    AI_TT_MAX_SIZE,
    AI_VCF_ENABLED,
    AI_VCF_MAX_CANDIDATES,
    AI_VCF_MAX_DEPTH,
    BOARD_SIZE,
    Player,
)

try:
    from gomoku.ai._threat_kernels import analyze_move as _analyze_move_native
    from gomoku.ai._threat_kernels import analyze_moves as _analyze_moves_native
    from gomoku.ai._threat_kernels import local_hotness as _local_hotness_native
except ImportError:  # pragma: no cover - exercised when extension is not built
    _analyze_move_native = None
    _analyze_moves_native = None
    _local_hotness_native = None

# 置换表条目：(depth, score, flag, best_move)
# flag: "E"=exact, "L"=lower bound (V >= score), "U"=upper bound (V <= score)
_TTEntry = tuple[int, float, str, Optional[tuple[int, int]]]
_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (1, -1))
_MoveAnalysis = tuple[bool, int]
_NONE = int(Player.NONE)
_FORCED_SCORE = 100_000.0
_EARLY_ROOT_RERANK_MAX_PLY = 9
_EARLY_ROOT_RERANK_TOP_K = 6
_EARLY_ROOT_RERANK_REPLY_TOP_K = 3
_EARLY_ROOT_RERANK_STABILIZER_TOP_K = 3
_EARLY_ROOT_RERANK_LAMBDA_MAX = 0.8
_EARLY_ROOT_RERANK_LAMBDA_AVG = 0.2
_EARLY_ROOT_RERANK_ENABLED = True
_EARLY_ROOT_RERANK_BLACK_ENABLED = True
_EARLY_ROOT_RERANK_WHITE_ENABLED = True
_EARLY_ROOT_RERANK_WHITE_REPLY_TOP_K = 3


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


@dataclass
class DecisionTrace:
    """Lightweight trace for understanding which stage selected the move."""

    source: str = ""
    move: Optional[tuple[int, int]] = None
    completed_depth: int = 0
    score: Optional[float] = None
    root_candidates: list[dict[str, object]] | None = None
    notes: list[str] | None = None


class SearchTimeout(Exception):
    """Raised when iterative deepening exceeds the configured time budget."""


class AISearcher:
    """基于 Minimax + Alpha-Beta 剪枝的五子棋 AI 搜索器。

    Attributes:
        depth: 搜索深度上限；当前正式基线常用 `depth=5`。
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
        self._killers: dict[int, list[tuple[int, int]]] = {}
        self._eval_cache: dict[int, int] = {}
        self._vcf = VCFSolver(max_candidates=AI_VCF_MAX_CANDIDATES)
        self.last_search_stats = SearchStats()
        self.last_decision_trace = DecisionTrace()
        self._deadline: Optional[float] = None

    @staticmethod
    def _trace_move_dict(
        move: tuple[int, int],
        *,
        score: float | int | None = None,
        attack_score: int | None = None,
        defense_score: int | None = None,
        hotness: float | int | None = None,
        tag: str | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {"move": [move[0], move[1]]}
        if score is not None:
            payload["score"] = score
        if attack_score is not None:
            payload["attack_score"] = attack_score
        if defense_score is not None:
            payload["defense_score"] = defense_score
        if hotness is not None:
            payload["hotness"] = hotness
        if tag is not None:
            payload["tag"] = tag
        return payload

    def _root_hotness_snapshot(
        self,
        board: Board,
        current_player: Player,
        *,
        limit: int = AI_MAX_CANDIDATES,
    ) -> list[dict[str, object]]:
        moves = self._candidate_moves(board)
        ranked: list[tuple[float, int, int, int, int]] = []
        opponent = self._opponent_of(current_player)
        for row, col in moves:
            attack_score = self._local_hotness(board, row, col, current_player)
            defense_score = self._local_hotness(board, row, col, opponent)
            hotness = attack_score + defense_score * DEFENSE_WEIGHT
            ranked.append((hotness, row, col, attack_score, defense_score))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [
            self._trace_move_dict(
                (row, col),
                attack_score=attack_score,
                defense_score=defense_score,
                hotness=hotness,
            )
            for hotness, row, col, attack_score, defense_score in ranked[:limit]
        ]

    def _should_apply_early_root_rerank(self, board: Board) -> bool:
        return (
            _EARLY_ROOT_RERANK_ENABLED
            and self._is_early_root_rerank_enabled_for_player()
            and len(board.move_history) <= _EARLY_ROOT_RERANK_MAX_PLY
        )

    def _is_early_root_rerank_enabled_for_player(self) -> bool:
        if self.ai_player == Player.BLACK:
            return _EARLY_ROOT_RERANK_BLACK_ENABLED
        return _EARLY_ROOT_RERANK_WHITE_ENABLED

    def _early_root_rerank_reply_top_k(self) -> int:
        if self.ai_player == Player.WHITE:
            return _EARLY_ROOT_RERANK_WHITE_REPLY_TOP_K
        return _EARLY_ROOT_RERANK_REPLY_TOP_K

    def _probe_opponent_reply_score(
        self,
        board: Board,
        candidate_move: tuple[int, int],
    ) -> dict[str, object]:
        reply_top_k = self._early_root_rerank_reply_top_k()
        board.place(candidate_move[0], candidate_move[1], self.ai_player)
        try:
            reply_moves = self._candidate_moves(board)
            if not reply_moves:
                return {
                    "max_reply_score": 0.0,
                    "reply_candidates": [],
                }

            immediate_opponent_wins = self._find_immediate_winning_moves(
                board, reply_moves, self._opponent
            )
            if immediate_opponent_wins:
                replies = [
                    self._trace_move_dict(move, score=_FORCED_SCORE, tag="opponent_immediate_win")
                    for move in immediate_opponent_wins[:reply_top_k]
                ]
                return {
                    "max_reply_score": _FORCED_SCORE,
                    "reply_candidates": replies,
                }

            ordering_stats = SearchStats()
            ordered_replies = self._order_moves(
                board,
                reply_moves,
                self._opponent,
                1,
                None,
                ordering_stats,
                use_killers=False,
            )[:reply_top_k]

            reply_candidates: list[dict[str, object]] = []
            max_reply_score = -math.inf
            total_reply_score = 0.0
            for reply in ordered_replies:
                attack_score = self._local_hotness(board, reply[0], reply[1], self._opponent)
                defense_score = self._local_hotness(board, reply[0], reply[1], self.ai_player)
                board.place(reply[0], reply[1], self._opponent)
                try:
                    if board.check_win(reply[0], reply[1]):
                        reply_score = _FORCED_SCORE
                        stabilizer_candidates: list[dict[str, object]] = []
                    else:
                        stabilizer_moves = self._candidate_moves(board)
                        if not stabilizer_moves:
                            reply_score = float(-self._evaluate(board))
                            stabilizer_candidates = []
                        else:
                            ordering_stats = SearchStats()
                            ordered_stabilizers = self._order_moves(
                                board,
                                stabilizer_moves,
                                self.ai_player,
                                1,
                                None,
                                ordering_stats,
                                use_killers=False,
                            )[:_EARLY_ROOT_RERANK_STABILIZER_TOP_K]
                            best_white_reply_eval = -math.inf
                            stabilizer_candidates = []
                            for stabilizer in ordered_stabilizers:
                                board.place(stabilizer[0], stabilizer[1], self.ai_player)
                                try:
                                    if board.check_win(stabilizer[0], stabilizer[1]):
                                        white_reply_eval = _FORCED_SCORE
                                    else:
                                        white_reply_eval = self._probe_stabilizer_eval(board)
                                finally:
                                    board.undo()
                                best_white_reply_eval = max(best_white_reply_eval, white_reply_eval)
                                stabilizer_candidates.append(
                                    self._trace_move_dict(stabilizer, score=white_reply_eval)
                                )
                            if best_white_reply_eval == -math.inf:
                                best_white_reply_eval = float(self._evaluate(board))
                            # Convert back to black's perspective after white's best stabilizing reply.
                            reply_score = -best_white_reply_eval
                finally:
                    board.undo()

                max_reply_score = max(max_reply_score, reply_score)
                total_reply_score += reply_score
                reply_candidates.append(
                    self._trace_move_dict(
                        reply,
                        score=reply_score,
                        attack_score=attack_score,
                        defense_score=defense_score,
                        tag="black_reply_probe",
                    )
                )
                if stabilizer_candidates:
                    reply_candidates[-1]["stabilizer_candidates"] = stabilizer_candidates

            if max_reply_score == -math.inf:
                max_reply_score = 0.0
            avg_reply_score = total_reply_score / len(reply_candidates) if reply_candidates else 0.0

            return {
                "max_reply_score": max_reply_score,
                "avg_reply_score": avg_reply_score,
                "reply_candidates": reply_candidates,
            }
        finally:
            board.undo()

    def _probe_stabilizer_eval(self, board: Board) -> float:
        """Evaluate a stabilizing reply for rerank probing.

        Rerank was overly optimistic when a stabilizer created a large static
        score (for example an OPEN_FOUR) but still allowed the opponent an
        immediate winning move on the next ply. Guard that case first so the
        probe does not treat tactically lost lines as highly favorable.
        """
        opponent_moves = self._candidate_moves(board)
        if opponent_moves:
            opponent_immediate_wins = self._find_immediate_winning_moves(
                board, opponent_moves, self._opponent
            )
            if opponent_immediate_wins:
                return float(-_FORCED_SCORE)
        return float(self._evaluate(board))

    def _rerank_early_root_candidates(
        self,
        board: Board,
        root_candidates: list[dict[str, object]],
    ) -> tuple[Optional[tuple[int, int]], Optional[float], list[dict[str, object]]]:
        ranked = [dict(candidate) for candidate in root_candidates]
        top = ranked[:_EARLY_ROOT_RERANK_TOP_K]
        for idx, candidate in enumerate(top, start=1):
            move_data = candidate.get("move")
            if not isinstance(move_data, list) or len(move_data) != 2:
                continue
            move = (int(move_data[0]), int(move_data[1]))
            base_score = float(candidate.get("score", -math.inf))
            probe = self._probe_opponent_reply_score(board, move)
            max_reply_score = float(probe["max_reply_score"])
            avg_reply_score = float(probe.get("avg_reply_score", max_reply_score))
            rerank_score = (
                base_score
                - _EARLY_ROOT_RERANK_LAMBDA_MAX * max_reply_score
                - _EARLY_ROOT_RERANK_LAMBDA_AVG * avg_reply_score
            )
            candidate["base_score"] = base_score
            candidate["max_reply_score"] = max_reply_score
            candidate["avg_reply_score"] = avg_reply_score
            candidate["rerank_score"] = rerank_score
            candidate["reply_candidates"] = probe["reply_candidates"]
            candidate["base_rank"] = idx

        top.sort(
            key=lambda item: (
                -float(item.get("rerank_score", item.get("score", -math.inf))),
                -float(item.get("score", -math.inf)),
                item["move"][0],
                item["move"][1],
            )
        )
        for idx, candidate in enumerate(top, start=1):
            candidate["rerank_rank"] = idx
        ranked[: len(top)] = top

        if not ranked:
            return None, None, ranked
        best = ranked[0]
        move = best.get("move")
        if not isinstance(move, list) or len(move) != 2:
            return None, None, ranked
        return (int(move[0]), int(move[1])), float(best.get("score", -math.inf)), ranked

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
        self.last_decision_trace = DecisionTrace()
        self._killers.clear()
        self._deadline = (
            time.perf_counter() + self.time_limit_s if self.time_limit_s is not None else None
        )

        moves = self._candidate_moves(board)
        if not moves:
            self.last_decision_trace = DecisionTrace(
                source="no_candidates",
                root_candidates=[],
            )
            self._deadline = None
            return None

        immediate_wins = self._find_immediate_winning_moves(board, moves, self.ai_player)
        if immediate_wins:
            self.last_search_stats.immediate_wins = 1
            self.last_search_stats.completed_depth = 1
            self.last_decision_trace = DecisionTrace(
                source="immediate_win",
                move=immediate_wins[0],
                completed_depth=1,
                root_candidates=[
                    self._trace_move_dict(move, score=100_000, tag="immediate_win")
                    for move in immediate_wins
                ],
            )
            self._deadline = None
            return immediate_wins[0]

        opponent_immediate_wins = self._find_immediate_winning_moves(board, moves, self._opponent)
        if opponent_immediate_wins:
            self.last_search_stats.completed_depth = 1
            self.last_decision_trace = DecisionTrace(
                source="immediate_block",
                move=opponent_immediate_wins[0],
                completed_depth=1,
                root_candidates=[
                    self._trace_move_dict(move, score=100_000, tag="opponent_immediate_win")
                    for move in opponent_immediate_wins
                ],
                notes=["blocked_opponent_immediate_win"],
            )
            self._deadline = None
            return opponent_immediate_wins[0]

        if AI_VCF_ENABLED:
            self._vcf.reset()

            vcf_move = self._vcf.find_winning_move(board, self.ai_player, AI_VCF_MAX_DEPTH)
            if vcf_move is not None:
                self.last_decision_trace = DecisionTrace(
                    source="vcf_win",
                    move=vcf_move,
                    completed_depth=AI_VCF_MAX_DEPTH,
                    root_candidates=self._root_hotness_snapshot(board, self.ai_player),
                    notes=[
                        f"vcf_nodes={self._vcf.last_stats.nodes}",
                        f"vcf_cache_hits={self._vcf.last_stats.cache_hits}",
                    ],
                )
                self._deadline = None
                return vcf_move

            vcf_block = self._vcf.find_blocking_move(board, self.ai_player, AI_VCF_MAX_DEPTH)
            if vcf_block is not None:
                self.last_decision_trace = DecisionTrace(
                    source="vcf_block",
                    move=vcf_block,
                    completed_depth=AI_VCF_MAX_DEPTH,
                    root_candidates=self._root_hotness_snapshot(board, self.ai_player),
                    notes=[
                        "blocked_opponent_vcf",
                        f"vcf_nodes={self._vcf.last_stats.nodes}",
                        f"vcf_cache_hits={self._vcf.last_stats.cache_hits}",
                    ],
                )
                self._deadline = None
                return vcf_block

        best_move: Optional[tuple[int, int]] = None
        best_score: Optional[float] = None
        best_root_candidates: list[dict[str, object]] | None = None
        for current_depth in range(1, self.depth + 1):
            iteration_stats = SearchStats()
            root_candidates: list[dict[str, object]] = []
            try:
                score, move = self._minimax(
                    board,
                    current_depth,
                    -math.inf,
                    math.inf,
                    maximizing=True,
                    stats=iteration_stats,
                    root_trace=root_candidates,
                )
            except SearchTimeout:
                self.last_search_stats.timed_out = True
                break

            iteration_stats.completed_depth = current_depth
            self.last_search_stats.merge(iteration_stats)
            best_move = move
            best_score = score
            best_root_candidates = root_candidates

            root_candidates.sort(
                key=lambda item: (-float(item.get("score", -math.inf)), item["move"][0], item["move"][1])
            )
            if self._should_apply_early_root_rerank(board) and root_candidates:
                reranked_move, reranked_score, reranked_candidates = self._rerank_early_root_candidates(
                    board, root_candidates
                )
                if reranked_move is not None and reranked_score is not None:
                    move = reranked_move
                    score = reranked_score
                    root_candidates = reranked_candidates

            if move is not None:
                best_move = move
            best_score = score
            best_root_candidates = root_candidates
            self._sync_root_tt_best_move(board, current_depth, best_move)

            if abs(score) >= _FORCED_SCORE:
                break

            if self._deadline is not None and time.perf_counter() >= self._deadline:
                self.last_search_stats.timed_out = True
                break

        self._deadline = None
        self.last_decision_trace.source = "minimax"
        self.last_decision_trace.move = best_move
        self.last_decision_trace.completed_depth = self.last_search_stats.completed_depth
        self.last_decision_trace.score = best_score
        self.last_decision_trace.root_candidates = best_root_candidates
        if (
            self._should_apply_early_root_rerank(board)
            and best_root_candidates
            and "rerank_score" in best_root_candidates[0]
        ):
            self.last_decision_trace.notes = [
                f"early_root_rerank_lambda_max={_EARLY_ROOT_RERANK_LAMBDA_MAX}",
                f"early_root_rerank_lambda_avg={_EARLY_ROOT_RERANK_LAMBDA_AVG}",
                f"early_root_rerank_top_k={_EARLY_ROOT_RERANK_TOP_K}",
                f"early_root_rerank_reply_top_k={_EARLY_ROOT_RERANK_REPLY_TOP_K}",
                f"early_root_rerank_stabilizer_top_k={_EARLY_ROOT_RERANK_STABILIZER_TOP_K}",
            ]
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

    def _sync_root_tt_best_move(
        self,
        board: Board,
        depth: int,
        best_move: Optional[tuple[int, int]],
    ) -> None:
        """Keep the root TT best move aligned with the final root decision."""
        if best_move is None:
            return
        entry = self._tt.get(board.hash)
        if entry is None:
            return
        tt_depth, tt_score, tt_flag, _ = entry
        if tt_depth != depth:
            return
        self._store_tt(self._tt, board.hash, (tt_depth, tt_score, tt_flag, best_move))

    def _check_timeout(self) -> None:
        """Raise when the configured time budget has been exhausted."""
        if self._deadline is not None and time.perf_counter() >= self._deadline:
            raise SearchTimeout

    @staticmethod
    def _line_tactical_score(length: int, open_ends: int) -> int:
        """对单方向连子形状给出轻量分值，用于候选排序。"""
        if length >= 5:
            return 200_000
        if length == 4:
            return 80_000 if open_ends == 2 else 30_000 if open_ends == 1 else 0
        if length == 3:
            return 8_000 if open_ends == 2 else 2_000 if open_ends == 1 else 0
        if length == 2:
            return 500 if open_ends == 2 else 100 if open_ends == 1 else 0
        if length == 1:
            return 30 if open_ends == 2 else 5 if open_ends == 1 else 0
        return 0

    @staticmethod
    def _count_one_side(
        board: Board,
        row: int,
        col: int,
        dr: int,
        dc: int,
        player: Player,
    ) -> tuple[int, bool]:
        """统计某一侧连续棋子数量，以及末端是否仍为空。"""
        grid = board.grid
        r, c = row + dr, col + dc
        length = 0
        while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == player:
            length += 1
            r += dr
            c += dc
        is_open = 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == _NONE
        return length, is_open

    def _analyze_move_for_player(
        self,
        board: Board,
        row: int,
        col: int,
        player: Player,
    ) -> _MoveAnalysis:
        """对单个空位做一次局部分析，复用给胜负判断和排序评分。"""
        if _analyze_move_native is not None:
            is_win, score = _analyze_move_native(board.grid, row, col, int(player))
            return bool(is_win), int(score)

        if board.grid[row, col] != _NONE:
            return False, -1

        directional_scores: list[int] = []
        for dr, dc in _DIRECTIONS:
            left_len, left_open = self._count_one_side(board, row, col, -dr, -dc, player)
            right_len, right_open = self._count_one_side(board, row, col, dr, dc, player)
            total_len = 1 + left_len + right_len
            if total_len >= 5:
                return True, 200_000
            open_ends = int(left_open) + int(right_open)
            directional_scores.append(self._line_tactical_score(total_len, open_ends))

        directional_scores.sort(reverse=True)
        total_score = sum(directional_scores)
        if len(directional_scores) >= 2:
            total_score += directional_scores[0] * directional_scores[1] // 20_000

        # 轻微偏向中心，减少同分时边缘点无意义靠前。
        center = board.grid.shape[0] // 2
        center_bias = 2 * board.grid.shape[0] - abs(row - center) - abs(col - center)
        return False, total_score + center_bias

    def _is_immediate_winning_move(
        self,
        board: Board,
        row: int,
        col: int,
        current_player: Player,
    ) -> bool:
        """仅基于落点周围四个方向，判断该空位是否一步成五。"""
        is_win, _ = self._analyze_move_for_player(board, row, col, current_player)
        return is_win

    def _find_immediate_winning_moves(
        self,
        board: Board,
        moves: list[tuple[int, int]],
        current_player: Player,
    ) -> list[tuple[int, int]]:
        """返回候选点中所有一步直接成五的着法。"""
        analyses = self._analyze_moves_for_player(board, moves, current_player)
        return [move for move, (is_win, _) in analyses.items() if is_win]

    def _analyze_moves_for_player(
        self,
        board: Board,
        moves: list[tuple[int, int]],
        player: Player,
    ) -> dict[tuple[int, int], _MoveAnalysis]:
        """批量分析当前层候选点，避免同一点重复扫描。"""
        if _analyze_moves_native is not None:
            analyses = _analyze_moves_native(board.grid, moves, int(player))
            return {move: (bool(is_win), int(score)) for move, (is_win, score) in zip(moves, analyses, strict=False)}
        return {
            (row, col): self._analyze_move_for_player(board, row, col, player)
            for row, col in moves
        }

    def _opponent_of(self, player: Player) -> Player:
        return Player.WHITE if player == Player.BLACK else Player.BLACK

    def _symmetry_move_key(self, board: Board, move: tuple[int, int]) -> tuple[int, int, int, int]:
        """Return a tie-break key that avoids absolute row/col direction bias."""
        row, col = move
        center = (BOARD_SIZE - 1) // 2
        center_dr = abs(row - center)
        center_dc = abs(col - center)
        last_row, last_col = board.last_move if board.last_move is not None else (center, center)
        last_dr = abs(row - last_row)
        last_dc = abs(col - last_col)
        return (
            max(last_dr, last_dc),
            last_dr + last_dc,
            max(center_dr, center_dc),
            center_dr + center_dc,
        )

    def _order_moves(
        self,
        board: Board,
        moves: list[tuple[int, int]],
        current_player: Player,
        depth: int,
        tt_move: Optional[tuple[int, int]],
        stats: SearchStats,
        *,
        use_killers: bool = True,
    ) -> list[tuple[int, int]]:
        """Order recursive moves by TT, killers, and local tactical hotness."""
        killers = set(self._killers.get(depth, [])) if use_killers else set()
        priority: list[tuple[int, int]] = []
        normal: list[tuple[int, int]] = []
        for move in moves:
            if tt_move is not None and move == tt_move:
                priority.insert(0, move)
            elif move in killers:
                priority.append(move)
            else:
                normal.append(move)

        scored: list[tuple[float, int, int]] = []
        for row, col in normal:
            stats.ordering_evals += 1
            attack_score = self._local_hotness(board, row, col, current_player)
            defense_score = self._local_hotness(board, row, col, self._opponent_of(current_player))
            hotness = attack_score + defense_score * DEFENSE_WEIGHT
            scored.append((hotness, row, col))

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        ordered = priority + [(row, col) for _, row, col in scored]
        return ordered[:AI_MAX_CANDIDATES]

    @staticmethod
    def _candidate_moves(board: Board) -> list[tuple[int, int]]:
        """Return search candidates from the board-owned candidate pool."""
        return board.get_candidate_moves()

    def _local_hotness(
        self,
        board: Board,
        row: int,
        col: int,
        player: Player,
    ) -> int:
        """Return the local ordering hotness for one candidate point."""
        if _local_hotness_native is not None:
            return int(_local_hotness_native(board.grid, row, col, int(player)))
        return self._local_hotness_python(board, row, col, player)

    def _local_hotness_python(
        self,
        board: Board,
        row: int,
        col: int,
        player: Player,
    ) -> int:
        """Pure Python baseline for local ordering hotness."""
        if board.grid[row, col] != _NONE:
            return 0

        score = 0
        for dr, dc in _DIRECTIONS:
            left_len, left_open = self._count_one_side(board, row, col, -dr, -dc, player)
            right_len, right_open = self._count_one_side(board, row, col, dr, dc, player)
            total_len = 1 + left_len + right_len
            open_ends = int(left_open) + int(right_open)
            score += self._line_tactical_score(total_len, open_ends)
        return score

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

    @classmethod
    def _prioritize_special_moves(
        cls,
        moves: list[tuple[int, int]],
        tt_move: Optional[tuple[int, int]],
        killer_moves: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Prioritize TT best move first, then killer moves for the same depth."""
        ordered = cls._prioritize_tt_move(moves, tt_move)
        if not killer_moves:
            return ordered

        killers_in_moves = [move for move in killer_moves if move in ordered and move != tt_move]
        if not killers_in_moves:
            return ordered

        front: list[tuple[int, int]] = []
        if tt_move is not None and tt_move in ordered:
            front.append(tt_move)
        front.extend(killers_in_moves)
        seen = set(front)
        tail = [move for move in ordered if move not in seen]
        return front + tail

    def _add_killer(self, depth: int, move: tuple[int, int]) -> None:
        killers = self._killers.setdefault(depth, [])
        if move in killers:
            killers.remove(move)
        killers.insert(0, move)
        if len(killers) > 2:
            killers.pop()

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
        root_trace: Optional[list[dict[str, object]]] = None,
    ) -> tuple[float, Optional[tuple[int, int]]]:
        """Minimax 递归搜索（Alpha-Beta 剪枝 + 置换表）。"""
        self._check_timeout()
        stats.nodes += 1
        h = board.hash
        alpha_orig = alpha

        entry = self._tt.get(h)
        tt_move = None
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

        if depth == 0:
            stats.leaf_evals += 1
            score = self._evaluate(board)
            return score, None

        moves = self._candidate_moves(board)
        if not moves:
            stats.leaf_evals += 1
            score = self._evaluate(board)
            return score, None

        current_player = self.ai_player if maximizing else self._opponent
        stats.max_branching = max(stats.max_branching, len(moves))
        moves = self._order_moves(board, moves, current_player, depth, tt_move, stats)

        best_move: Optional[tuple[int, int]] = None
        if maximizing:
            best_score = -math.inf
            for row, col in moves:
                self._check_timeout()
                board.place(row, col, current_player)
                try:
                    if board.check_win(row, col):
                        score = 100_000.0
                    else:
                        score, _ = self._minimax(
                            board,
                            depth - 1,
                            alpha,
                            beta,
                            False,
                            stats,
                            None,
                        )
                finally:
                    board.undo()

                if root_trace is not None:
                    root_trace.append(self._trace_move_dict((row, col), score=score))

                if score > best_score:
                    best_score = score
                    best_move = (row, col)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    stats.beta_cutoffs += 1
                    self._add_killer(depth, (row, col))
                    break

            flag = "U" if best_score <= alpha_orig else "E"
            if best_score >= beta:
                flag = "L"
            self._store_tt(self._tt, h, (depth, best_score, flag, best_move))
            return best_score, best_move

        best_score = math.inf
        for row, col in moves:
            self._check_timeout()
            board.place(row, col, current_player)
            try:
                if board.check_win(row, col):
                    score = -100_000.0
                else:
                    score, _ = self._minimax(
                        board,
                        depth - 1,
                        alpha,
                        beta,
                        True,
                        stats,
                        None,
                    )
            finally:
                board.undo()

            if score < best_score:
                best_score = score
                best_move = (row, col)
            beta = min(beta, best_score)
            if beta <= alpha:
                stats.alpha_cutoffs += 1
                self._add_killer(depth, (row, col))
                break

        flag = "L" if best_score >= beta else "E"
        if best_score <= alpha_orig:
            flag = "U"
        self._store_tt(self._tt, h, (depth, best_score, flag, best_move))
        return best_score, best_move
