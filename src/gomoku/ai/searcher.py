"""Minimax searcher with alpha-beta pruning and transposition table for Gomoku AI."""

import math
import time
from dataclasses import dataclass
from typing import Optional

from gomoku.ai.evaluator import evaluate
from gomoku.ai.threats import (
    ThreatType,
    classify_attack_moves,
    classify_defense_moves,
    classify_moves,
)
from gomoku.ai.vcf import VCFSolver
from gomoku.board import Board
from gomoku.config import (
    AI_EVAL_CACHE_MAX_SIZE,
    AI_MAX_CANDIDATES,
    AI_QUIESCENCE_MAX_CANDIDATES,
    AI_QUIESCENCE_MAX_PLY,
    AI_ROOT_RERANK_TOP_K,
    AI_TT_MAX_SIZE,
    AI_VCF_ENABLED,
    AI_VCF_MAX_CANDIDATES,
    AI_VCF_MAX_DEPTH,
    Player,
)

try:
    from gomoku.ai._threat_kernels import analyze_move as _analyze_move_native
    from gomoku.ai._threat_kernels import analyze_moves as _analyze_moves_native
except ImportError:  # pragma: no cover - exercised when extension is not built
    _analyze_move_native = None
    _analyze_moves_native = None

# 置换表条目：(depth, score, flag, best_move)
# flag: "E"=exact, "L"=lower bound (V >= score), "U"=upper bound (V <= score)
_TTEntry = tuple[int, float, str, Optional[tuple[int, int]]]
_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (1, -1))
_ORDERING_RERANK_TOP_K = 8
_MoveAnalysis = tuple[bool, int]
_FORCING_SEARCH_DEPTH = 4
_FORCING_SEARCH_ENABLED = False
_QUIESCENCE_THREATS: tuple[ThreatType, ...] = (
    ThreatType.WIN,
    ThreatType.OPEN_FOUR,
    ThreatType.DOUBLE_HALF_FOUR,
)


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
    forcing_wins: int = 0
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
        self.forcing_wins += other.forcing_wins
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
        self._threat_cache: dict[tuple[int, int, str], list] = {}
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

        moves = board.get_candidate_moves()
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

        if _FORCING_SEARCH_ENABLED:
            forcing_move = self._find_forcing_move(board, self.ai_player)
            if forcing_move is not None:
                self.last_search_stats.forcing_wins = 1
                self.last_search_stats.completed_depth = 1
                self.last_decision_trace = DecisionTrace(
                    source="forcing_search",
                    move=forcing_move,
                    completed_depth=1,
                )
                self._deadline = None
                return forcing_move

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
                    root_depth=current_depth,
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

    def _classify_moves_cached(
        self,
        board: Board,
        moves: list[tuple[int, int]],
        player: Player,
        mode: str = "both",
    ) -> list:
        """Cache threat classification per board hash and side to avoid recomputation."""
        key = (board.hash, int(player), mode)
        cached = self._threat_cache.get(key)
        if cached is not None:
            return cached
        if mode == "attack":
            classified = classify_attack_moves(board, moves, player)
        elif mode == "defense":
            classified = classify_defense_moves(board, moves, player)
        else:
            classified = classify_moves(board, moves, player)
        if len(self._threat_cache) >= AI_EVAL_CACHE_MAX_SIZE:
            self._threat_cache.clear()
        self._threat_cache[key] = classified
        return classified

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

    def _score_ordering_move(
        self,
        board: Board,
        row: int,
        col: int,
        current_player: Player,
    ) -> tuple[int, int, int]:
        """返回综合排序分，以及进攻/防守分量。"""
        opponent = self.ai_player if current_player == self._opponent else self._opponent
        _, attack_score = self._analyze_move_for_player(board, row, col, current_player)
        _, defense_score = self._analyze_move_for_player(board, row, col, opponent)
        return attack_score * 2 + defense_score * 3, attack_score, defense_score

    @staticmethod
    def _forcing_move_priority(threat_type: ThreatType) -> int:
        if threat_type == ThreatType.WIN:
            return 8
        if threat_type == ThreatType.BLOCK_WIN:
            return 7
        if threat_type == ThreatType.OPEN_FOUR:
            return 6
        if threat_type == ThreatType.DOUBLE_HALF_FOUR:
            return 5
        if threat_type == ThreatType.FOUR_THREE:
            return 4
        if threat_type == ThreatType.DOUBLE_OPEN_THREE:
            return 3
        if threat_type == ThreatType.HALF_FOUR:
            return 2
        if threat_type == ThreatType.OPEN_THREE:
            return 1
        return 0

    def _forcing_candidates(
        self,
        board: Board,
        player: Player,
        attacker: Player,
    ) -> list[tuple[tuple[int, int], ThreatType]]:
        """Return threat-ranked candidates for forcing search."""
        mode = "attack" if player == attacker else "defense"
        infos = self._classify_moves_cached(board, board.get_candidate_moves(), player, mode=mode)
        if player == attacker:
            allowed = {
                ThreatType.WIN,
                ThreatType.OPEN_FOUR,
                ThreatType.DOUBLE_HALF_FOUR,
                ThreatType.HALF_FOUR,
                ThreatType.DOUBLE_OPEN_THREE,
                ThreatType.FOUR_THREE,
                ThreatType.OPEN_THREE,
            }
            candidates = [
                (info.move, info.attack_type) for info in infos if info.attack_type in allowed
            ]
        else:
            allowed = {
                ThreatType.WIN,
                ThreatType.OPEN_FOUR,
                ThreatType.DOUBLE_HALF_FOUR,
                ThreatType.HALF_FOUR,
                ThreatType.DOUBLE_OPEN_THREE,
                ThreatType.FOUR_THREE,
                ThreatType.OPEN_THREE,
            }
            all_candidates = [
                (info.move, info.defense_type) for info in infos if info.defense_type in allowed
            ]
            if not all_candidates:
                return []
            strongest = max(
                self._forcing_move_priority(threat_type)
                for _, threat_type in all_candidates
            )
            candidates = [
                (move, threat_type)
                for move, threat_type in all_candidates
                if self._forcing_move_priority(threat_type) == strongest
            ]
        candidates.sort(
            key=lambda item: (self._forcing_move_priority(item[1]), item[0][0], item[0][1]),
            reverse=True,
        )
        return candidates

    def _forcing_search(
        self,
        board: Board,
        attacker: Player,
        current_player: Player,
        depth: int,
    ) -> tuple[bool, Optional[tuple[int, int]]]:
        """Search a short forcing line among high-priority threat moves only."""
        self._check_timeout()
        if depth <= 0:
            return False, None

        candidates = self._forcing_candidates(board, current_player, attacker)
        if not candidates:
            return (current_player != attacker), None

        if current_player == attacker:
            for move, threat_type in candidates:
                if threat_type == ThreatType.WIN:
                    return True, move
                board.place(*move, current_player)
                try:
                    success, _ = self._forcing_search(
                        board,
                        attacker,
                        self._opponent_of(current_player),
                        depth - 1,
                    )
                finally:
                    board.undo()
                if success:
                    return True, move
            return False, None

        for move, threat_type in candidates:
            board.place(*move, current_player)
            try:
                success, _ = self._forcing_search(board, attacker, attacker, depth - 1)
            finally:
                board.undo()
            if not success:
                return False, None
        return True, None

    def _opponent_of(self, player: Player) -> Player:
        return Player.WHITE if player == Player.BLACK else Player.BLACK

    def _find_forcing_move(
        self,
        board: Board,
        current_player: Player,
    ) -> Optional[tuple[int, int]]:
        """Try to prove a short forcing win from the current position."""
        success, move = self._forcing_search(
            board, current_player, current_player, _FORCING_SEARCH_DEPTH
        )
        return move if success else None

    @staticmethod
    def _dynamic_cutoff(
        scored_moves: list[tuple[int, int, int, int, int]],
        max_candidates: int,
    ) -> int:
        """按局面强度动态决定截断宽度。"""
        if not scored_moves:
            return 0

        limit = min(len(scored_moves), max_candidates)
        top_score = scored_moves[0][2]
        if limit <= 4:
            return limit

        if top_score >= 150_000:
            cutoff = 4
        elif top_score >= 60_000:
            cutoff = 6
        elif top_score >= 12_000:
            cutoff = 8
        elif top_score >= 2_000:
            cutoff = 10
        else:
            cutoff = limit

        cutoff = min(cutoff, limit)
        if cutoff >= limit:
            return limit

        gap = max(500, top_score // 5)
        while cutoff < limit and top_score - scored_moves[cutoff][2] <= gap:
            cutoff += 1
        return cutoff

    def _select_quiescence_moves(
        self,
        board: Board,
        current_player: Player,
        tt_move: Optional[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Return only high-priority tactical moves for quiescence search."""
        moves = board.get_candidate_moves()
        if not moves:
            return []

        killer_moves = self._killers.get(0, [])

        def _cap_quiescence_moves(candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
            if AI_QUIESCENCE_MAX_CANDIDATES <= 0:
                return []
            if len(candidates) <= AI_QUIESCENCE_MAX_CANDIDATES:
                return candidates
            return candidates[:AI_QUIESCENCE_MAX_CANDIDATES]

        current_move_analysis = self._analyze_moves_for_player(board, moves, current_player)
        immediate_wins = [move for move, (is_win, _) in current_move_analysis.items() if is_win]
        if immediate_wins:
            immediate_wins.sort()
            prioritized = self._prioritize_special_moves(immediate_wins, tt_move, killer_moves)
            return _cap_quiescence_moves(prioritized)

        opponent = self._opponent_of(current_player)
        opponent_move_analysis = self._analyze_moves_for_player(board, moves, opponent)
        blocking_moves = [move for move, (is_win, _) in opponent_move_analysis.items() if is_win]
        if blocking_moves:
            blocking_moves.sort()
            prioritized = self._prioritize_special_moves(blocking_moves, tt_move, killer_moves)
            return _cap_quiescence_moves(prioritized)

        defense_grouped: dict[ThreatType, list[tuple[int, int]]] = {
            threat_type: [] for threat_type in ThreatType
        }
        for info in self._classify_moves_cached(board, moves, current_player, mode="defense"):
            defense_grouped[info.defense_type].append(info.move)

        for threat_type in _QUIESCENCE_THREATS[1:]:
            threat_moves = defense_grouped[threat_type]
            if threat_moves:
                threat_moves.sort()
                prioritized = self._prioritize_special_moves(threat_moves, tt_move, killer_moves)
                return _cap_quiescence_moves(prioritized)

        attack_grouped: dict[ThreatType, list[tuple[int, int]]] = {
            threat_type: [] for threat_type in ThreatType
        }
        for info in self._classify_moves_cached(board, moves, current_player, mode="attack"):
            attack_grouped[info.attack_type].append(info.move)

        for threat_type in _QUIESCENCE_THREATS[1:]:
            threat_moves = attack_grouped[threat_type]
            if threat_moves:
                threat_moves.sort()
                prioritized = self._prioritize_special_moves(threat_moves, tt_move, killer_moves)
                return _cap_quiescence_moves(prioritized)

        return []

    def _quiescence(
        self,
        board: Board,
        alpha: float,
        beta: float,
        maximizing: bool,
        stats: SearchStats,
        remaining_ply: int,
    ) -> tuple[float, Optional[tuple[int, int]]]:
        """Extend leaf nodes along forcing tactical moves until the position is quiet."""
        self._check_timeout()
        stats.leaf_evals += 1
        stand_pat = self._evaluate(board)
        if remaining_ply <= 0:
            return stand_pat, None

        current_player = self.ai_player if maximizing else self._opponent
        tt_move = None
        entry = self._tt.get(board.hash)
        if entry is not None:
            tt_move = entry[3]
        moves = self._select_quiescence_moves(board, current_player, tt_move)
        if not moves:
            return stand_pat, None

        best_move: Optional[tuple[int, int]] = None
        if maximizing:
            best_score = stand_pat
            alpha = max(alpha, best_score)
            if beta <= alpha:
                return best_score, None
            for row, col in moves:
                self._check_timeout()
                board.place(row, col, current_player)
                try:
                    score, _ = self._quiescence(
                        board,
                        alpha,
                        beta,
                        False,
                        stats,
                        remaining_ply - 1,
                    )
                finally:
                    board.undo()
                if score > best_score:
                    best_score = score
                    best_move = (row, col)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score, best_move

        best_score = stand_pat
        beta = min(beta, best_score)
        if beta <= alpha:
            return best_score, None
        for row, col in moves:
            self._check_timeout()
            board.place(row, col, current_player)
            try:
                score, _ = self._quiescence(
                    board,
                    alpha,
                    beta,
                    True,
                    stats,
                    remaining_ply - 1,
                )
            finally:
                board.undo()
            if score < best_score:
                best_score = score
                best_move = (row, col)
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score, best_move

    def _select_search_moves(
        self,
        board: Board,
        moves: list[tuple[int, int]],
        current_player: Player,
        tt_move: Optional[tuple[int, int]],
        stats: SearchStats,
        depth: int = 0,
        rerank_top_k: int = _ORDERING_RERANK_TOP_K,
        current_move_analysis: Optional[dict[tuple[int, int], _MoveAnalysis]] = None,
        opponent_move_analysis: Optional[dict[tuple[int, int], _MoveAnalysis]] = None,
    ) -> list[tuple[int, int]]:
        """分层生成候选点，并对普通局面执行动态截断。"""
        killer_moves = self._killers.get(depth, [])
        defense_grouped: dict[ThreatType, list[tuple[int, int]]] = {
            threat_type: [] for threat_type in ThreatType
        }
        for info in self._classify_moves_cached(board, moves, current_player, mode="defense"):
            defense_grouped[info.defense_type].append(info.move)

        for threat_type in (
            ThreatType.WIN,
            ThreatType.OPEN_FOUR,
            ThreatType.DOUBLE_HALF_FOUR,
            ThreatType.FOUR_THREE,
            ThreatType.DOUBLE_OPEN_THREE,
        ):
            threat_moves = defense_grouped[threat_type]
            if threat_moves:
                threat_moves.sort()
                return self._prioritize_special_moves(threat_moves, tt_move, killer_moves)

        attack_grouped: dict[ThreatType, list[tuple[int, int]]] = {
            threat_type: [] for threat_type in ThreatType
        }
        for info in self._classify_moves_cached(board, moves, current_player, mode="attack"):
            attack_grouped[info.attack_type].append(info.move)

        for threat_type in (
            ThreatType.WIN,
            ThreatType.OPEN_FOUR,
            ThreatType.DOUBLE_HALF_FOUR,
            ThreatType.FOUR_THREE,
            ThreatType.DOUBLE_OPEN_THREE,
        ):
            threat_moves = attack_grouped[threat_type]
            if threat_moves:
                threat_moves.sort()
                return self._prioritize_special_moves(threat_moves, tt_move, killer_moves)

        opponent = self.ai_player if current_player == self._opponent else self._opponent
        if opponent_move_analysis is None:
            opponent_move_analysis = self._analyze_moves_for_player(board, moves, opponent)
        blocking_moves = [move for move, (is_win, _) in opponent_move_analysis.items() if is_win]
        if blocking_moves:
            ordered_blocks = sorted(blocking_moves)
            return self._prioritize_special_moves(ordered_blocks, tt_move, killer_moves)

        if current_move_analysis is None:
            current_move_analysis = self._analyze_moves_for_player(board, moves, current_player)

        scored_moves: list[tuple[int, int, int, int, int]] = []
        for r, c in moves:
            self._check_timeout()
            stats.ordering_evals += 1
            _, defense_score = opponent_move_analysis[(r, c)]
            is_attacking_win, attack_score = current_move_analysis[(r, c)]
            if is_attacking_win:
                attack_score = 200_000
            total_score = attack_score * 2 + defense_score * 3
            scored_moves.append((r, c, total_score, attack_score, defense_score))

        scored_moves.sort(key=lambda x: (x[2], x[3], x[4], x[0], x[1]), reverse=True)

        cutoff = self._dynamic_cutoff(scored_moves, AI_MAX_CANDIDATES)
        selected_moves = scored_moves[:cutoff]
        rerank_limit = min(len(selected_moves), rerank_top_k)
        reranked = self._rerank_top_moves(
            board, selected_moves, rerank_limit, current_player, stats
        )
        return self._prioritize_special_moves(
            [(r, c) for r, c, _, _, _ in reranked], tt_move, killer_moves
        )

    def _rerank_top_moves(
        self,
        board: Board,
        scored_moves: list[tuple[int, int, int, int, int]],
        rerank_limit: int,
        current_player: Player,
        stats: SearchStats,
    ) -> list[tuple[int, int, int, int, int]]:
        """对粗排前若干候选做完整局面评估，提升排序质量。"""
        if rerank_limit <= 1:
            return scored_moves

        reranked_prefix: list[tuple[int, int, int, int, int]] = []
        for row, col, coarse_score, attack_score, defense_score in scored_moves[:rerank_limit]:
            self._check_timeout()
            board.place(row, col, current_player)
            try:
                exact_score = self._evaluate(board)
            finally:
                board.undo()
            reranked_prefix.append((row, col, exact_score, attack_score, defense_score))

        reranked_prefix.sort(key=lambda x: (x[2], x[3], x[4], x[0], x[1]), reverse=True)
        return reranked_prefix + scored_moves[rerank_limit:]

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
        root_depth: int,
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
            score, move = self._quiescence(
                board,
                alpha,
                beta,
                maximizing,
                stats,
                AI_QUIESCENCE_MAX_PLY,
            )
            self._store_tt(self._tt, h, (0, score, "E", move))
            return score, move

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
        current_move_analysis = self._analyze_moves_for_player(board, moves, current_player)
        winning_moves = [move for move, (is_win, _) in current_move_analysis.items() if is_win]
        if winning_moves:
            stats.immediate_wins += 1
            winning_move = winning_moves[0]
            score = 100_000.0 if maximizing else -100_000.0
            self._store_tt(self._tt, h, (depth, score, "E", winning_move))
            return score, winning_move

        opponent = self._opponent_of(current_player)
        opponent_move_analysis = self._analyze_moves_for_player(board, moves, opponent)
        opponent_winning_moves = [move for move, (is_win, _) in opponent_move_analysis.items() if is_win]

        if not opponent_winning_moves:
            forcing_move = self._find_forcing_move(board, current_player)
            if forcing_move is not None:
                stats.forcing_wins += 1
                score = 90_000.0 if maximizing else -90_000.0
                self._store_tt(self._tt, h, (depth, score, "E", forcing_move))
                return score, forcing_move

        # --- 威胁驱动候选生成 + 动态截断 ---
        moves = self._select_search_moves(
            board,
            moves,
            current_player,
            tt_move,
            stats,
            depth=depth,
            rerank_top_k=(
                AI_ROOT_RERANK_TOP_K
                if maximizing and depth == root_depth
                else _ORDERING_RERANK_TOP_K
            ),
            current_move_analysis=current_move_analysis,
            opponent_move_analysis=opponent_move_analysis,
        )
        if maximizing and depth == root_depth:
            self.last_decision_trace.root_candidates = moves[:]

        best_move: Optional[tuple[int, int]] = None

        if maximizing:
            best_score: float = -math.inf
            for row, col in moves:
                self._check_timeout()
                board.place(row, col, current_player)
                try:
                    score, _ = self._minimax(
                        board,
                        depth - 1,
                        alpha,
                        beta,
                        False,
                        stats,
                        root_depth,
                    )
                finally:
                    board.undo()
                if score > best_score:
                    best_score = score
                    best_move = (row, col)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    # beta 截断：真实值 >= best_score，存为下界
                    stats.beta_cutoffs += 1
                    self._add_killer(depth, (row, col))
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
                    score, _ = self._minimax(
                        board,
                        depth - 1,
                        alpha,
                        beta,
                        True,
                        stats,
                        root_depth,
                    )
                finally:
                    board.undo()
                if score < best_score:
                    best_score = score
                    best_move = (row, col)
                beta = min(beta, best_score)
                if beta <= alpha:
                    # alpha 截断：真实值 <= best_score，存为上界
                    stats.alpha_cutoffs += 1
                    self._add_killer(depth, (row, col))
                    self._store_tt(self._tt, h, (depth, best_score, "U", best_move))
                    return best_score, best_move
            # 无截断：所有候选点均已遍历，best_score 是精确最小值
            self._store_tt(self._tt, h, (depth, best_score, "E", best_move))
            return best_score, best_move
