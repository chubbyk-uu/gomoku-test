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
    from gomoku.ai._threat_kernels import candidate_moves_radius1 as _candidate_moves_radius1_native
    from gomoku.ai._threat_kernels import local_hotness as _local_hotness_native
except ImportError:  # pragma: no cover - exercised when extension is not built
    _analyze_move_native = None
    _analyze_moves_native = None
    _candidate_moves_radius1_native = None
    _local_hotness_native = None

# 置换表条目：(depth, score, flag, best_move)
# flag: "E"=exact, "L"=lower bound (V >= score), "U"=upper bound (V <= score)
_TTEntry = tuple[int, float, str, Optional[tuple[int, int]]]
_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (1, -1))
_MoveAnalysis = tuple[bool, int]


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
    root_candidates: list[tuple[int, int]] | None = None
    notes: list[str] | None = None


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
        self._killers: dict[int, list[tuple[int, int]]] = {}
        self._eval_cache: dict[int, int] = {}
        self._vcf = VCFSolver(max_candidates=AI_VCF_MAX_CANDIDATES)
        self.last_search_stats = SearchStats()
        self.last_decision_trace = DecisionTrace()
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
                root_candidates=immediate_wins,
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
                root_candidates=opponent_immediate_wins,
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
                    notes=[
                        f"vcf_nodes={self._vcf.last_stats.nodes}",
                        f"vcf_cache_hits={self._vcf.last_stats.cache_hits}",
                    ],
                )
                self._deadline = None
                return vcf_block

        best_move: Optional[tuple[int, int]] = None
        best_score: Optional[float] = None
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
            best_score = score

            if abs(score) >= 100_000.0:
                break

            if self._deadline is not None and time.perf_counter() >= self._deadline:
                self.last_search_stats.timed_out = True
                break

        self._deadline = None
        self.last_decision_trace.source = "minimax"
        self.last_decision_trace.move = best_move
        self.last_decision_trace.completed_depth = self.last_search_stats.completed_depth
        self.last_decision_trace.score = best_score
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
        is_open = 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == Player.NONE
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

        if board.grid[row, col] != Player.NONE:
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
    ) -> list[tuple[int, int]]:
        """Order recursive moves by TT, killers, and local tactical hotness."""
        killers = set(self._killers.get(depth, []))
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
        """Return radius-1 row-major candidates for the current single search pipeline."""
        if _candidate_moves_radius1_native is not None:
            return _candidate_moves_radius1_native(board.grid)
        return AISearcher._candidate_moves_python(board)

    @staticmethod
    def _candidate_moves_python(board: Board) -> list[tuple[int, int]]:
        """Pure Python baseline for radius-1 row-major candidate generation."""
        if not board.move_history:
            center = BOARD_SIZE // 2
            return [(center, center)]

        moves: list[tuple[int, int]] = []
        grid = board.grid
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if grid[row, col] != Player.NONE:
                    continue
                found = False
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = row + dr, col + dc
                        if (
                            0 <= nr < BOARD_SIZE
                            and 0 <= nc < BOARD_SIZE
                            and grid[nr, nc] != Player.NONE
                        ):
                            found = True
                            break
                    if found:
                        break
                if found:
                    moves.append((row, col))
        return moves

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
        if board.grid[row, col] != Player.NONE:
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
                        score, _ = self._minimax(board, depth - 1, alpha, beta, False, stats)
                finally:
                    board.undo()

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
                    score, _ = self._minimax(board, depth - 1, alpha, beta, True, stats)
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
