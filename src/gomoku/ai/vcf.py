"""VCF (Victory by Continuous Fours) tactical solver.

The Python implementation is intentionally structured as a thin strategy layer
around small helper functions so that hot paths can later be moved into Cython
without changing the public interface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from gomoku.ai.threats import ThreatType, classify_attack_moves
from gomoku.board import Board
from gomoku.config import AI_VCF_MAX_CANDIDATES, Player

try:
    from gomoku.ai._threat_kernels import analyze_moves as _analyze_moves_native
    from gomoku.ai._threat_kernels import vcf_move_probes as _vcf_move_probes_native
except ImportError:  # pragma: no cover - exercised when extension is not built
    _analyze_moves_native = None
    _vcf_move_probes_native = None

Move = tuple[int, int]


@dataclass
class VCFStats:
    """Per-search profiling counters for the standalone VCF solver."""

    mode: str = ""
    elapsed_s: float = 0.0
    nodes: int = 0
    cache_hits: int = 0
    attack_candidates: int = 0
    defense_candidates: int = 0
    prefiltered_moves: int = 0
    classified_moves: int = 0
    immediate_win_checks: int = 0
    max_depth_reached: int = 0


class VCFSolver:
    """Standalone VCF solver with per-search recursion cache."""

    _ATTACK_PRIORITIES: dict[ThreatType, int] = {
        ThreatType.WIN: 5,
        ThreatType.OPEN_FOUR: 4,
        ThreatType.DOUBLE_HALF_FOUR: 3,
        ThreatType.HALF_FOUR: 2,
        ThreatType.FOUR_THREE: 1,
    }

    def __init__(self, max_candidates: int = AI_VCF_MAX_CANDIDATES) -> None:
        self.max_candidates = max_candidates
        self._tt: dict[tuple[int, int, int], Optional[Move]] = {}
        self.last_stats = VCFStats()
        self.last_trace: dict[str, object] | None = None

    def reset(self) -> None:
        """Reset per-search caches and stats."""
        self._tt.clear()
        self.last_stats = VCFStats()
        self.last_trace = None

    def find_winning_move(
        self,
        board: Board,
        attacker: Player,
        max_depth: int,
    ) -> Optional[Move]:
        """Return a VCF winning move for ``attacker`` if one can be proven."""
        self.reset()
        self.last_stats.mode = "win"
        self.last_trace = {
            "mode": "win",
            "actor": attacker.name,
            "max_depth": max_depth,
        }
        started_at = time.perf_counter()
        move = self._find_vcf_move(board, attacker, max_depth)
        self.last_stats.elapsed_s = time.perf_counter() - started_at
        if self.last_trace is not None:
            self.last_trace["selected_move"] = list(move) if move is not None else None
        return move

    def find_blocking_move(
        self,
        board: Board,
        defender: Player,
        max_depth: int,
    ) -> Optional[Move]:
        """Return one move that breaks the opponent's VCF if any."""
        self.reset()
        self.last_stats.mode = "block"
        started_at = time.perf_counter()
        attacker = self._opponent_of(defender)
        self.last_trace = {
            "mode": "block",
            "actor": defender.name,
            "attacker": attacker.name,
            "max_depth": max_depth,
            "defense_candidates": [],
            "defense_checks": [],
        }
        if self._has_vcf(board, attacker, max_depth) is None:
            self.last_stats.elapsed_s = time.perf_counter() - started_at
            if self.last_trace is not None:
                self.last_trace["selected_move"] = None
            return None

        defenses = self._generate_blocking_moves(board, defender)
        self.last_stats.defense_candidates += len(defenses)
        if self.last_trace is not None:
            self.last_trace["defense_candidates"] = [list(move) for move in defenses]
        for row, col in defenses:
            board.place(row, col, defender)
            try:
                if board.check_win(row, col):
                    if self.last_trace is not None:
                        checks = self.last_trace["defense_checks"]
                        assert isinstance(checks, list)
                        checks.append(
                            {
                                "move": [row, col],
                                "wins_immediately": True,
                                "remaining_vcf_move": None,
                                "accepted": True,
                            }
                        )
                        self.last_trace["selected_move"] = [row, col]
                    self.last_stats.elapsed_s = time.perf_counter() - started_at
                    return (row, col)
                self._tt.clear()
                remaining_vcf = self._has_vcf(board, attacker, max(max_depth - 1, 0))
                accepted = remaining_vcf is None
                if self.last_trace is not None:
                    checks = self.last_trace["defense_checks"]
                    assert isinstance(checks, list)
                    checks.append(
                        {
                            "move": [row, col],
                            "wins_immediately": False,
                            "remaining_vcf_move": (
                                None if remaining_vcf is None else [remaining_vcf[0], remaining_vcf[1]]
                            ),
                            "accepted": accepted,
                        }
                    )
                if accepted:
                    if self.last_trace is not None:
                        self.last_trace["selected_move"] = [row, col]
                    self.last_stats.elapsed_s = time.perf_counter() - started_at
                    return (row, col)
            finally:
                board.undo()
        self.last_stats.elapsed_s = time.perf_counter() - started_at
        if self.last_trace is not None:
            self.last_trace["selected_move"] = None
        return None

    def _find_vcf_move(
        self,
        board: Board,
        attacker: Player,
        depth: int,
    ) -> Optional[Move]:
        if depth <= 0:
            return None
        self.last_stats.max_depth_reached = max(self.last_stats.max_depth_reached, depth)
        for move in self._generate_vcf_attacks(board, attacker):
            if self._vcf_move_wins(board, attacker, move, depth):
                return move
        return None

    def _has_vcf(
        self,
        board: Board,
        attacker: Player,
        depth: int,
    ) -> Optional[Move]:
        if depth <= 0:
            return None

        self.last_stats.nodes += 1
        key = (board.hash, int(attacker), depth)
        if key in self._tt:
            self.last_stats.cache_hits += 1
            return self._tt[key]

        move = self._find_vcf_move(board, attacker, depth)
        self._tt[key] = move
        return move

    def _vcf_move_wins(
        self,
        board: Board,
        attacker: Player,
        move: Move,
        depth: int,
    ) -> bool:
        if depth <= 0:
            return False

        defender = self._opponent_of(attacker)
        row, col = move
        board.place(row, col, attacker)
        try:
            if board.check_win(row, col):
                return True

            if self._find_immediate_wins(board, defender, limit=1):
                return False

            defenses = self._generate_forced_defenses(board, attacker, defender)
            if not defenses:
                return False

            for defend_row, defend_col in defenses:
                board.place(defend_row, defend_col, defender)
                try:
                    if board.check_win(defend_row, defend_col):
                        return False
                    if self._has_vcf(board, attacker, depth - 1) is None:
                        return False
                finally:
                    board.undo()
            return True
        finally:
            board.undo()

    def _generate_vcf_attacks(
        self,
        board: Board,
        attacker: Player,
    ) -> list[Move]:
        strong_attacks = self._classify_strong_attacks(board, attacker)
        forcing: list[Move] = []

        for move, _priority, _score in strong_attacks:
            row, col = move
            board.place(row, col, attacker)
            try:
                if board.check_win(row, col) or self._find_immediate_wins(
                    board, attacker, limit=1
                ):
                    forcing.append(move)
            finally:
                board.undo()

            if len(forcing) >= self.max_candidates:
                break

        self.last_stats.attack_candidates += len(forcing)
        if self.last_trace is not None and "top_level_attacks" not in self.last_trace:
            self.last_trace["top_level_attacks"] = {
                "forcing_moves": [list(move) for move in forcing],
                "max_candidates": self.max_candidates,
            }
        return forcing

    def _generate_forced_defenses(
        self,
        board: Board,
        attacker: Player,
        defender: Player,
    ) -> list[Move]:
        """Return the minimal defense set after the attacker just played.

        In a strict VCF line, the defender should only be allowed to answer the
        attacker's immediate winning continuations. Broader tactical defenses are
        intentionally left to forcing search / minimax.
        """
        winning_responses = self._find_immediate_wins(board, attacker)
        if winning_responses:
            return winning_responses
        return []

    def _generate_blocking_moves(
        self,
        board: Board,
        defender: Player,
    ) -> list[Move]:
        attacker = self._opponent_of(defender)
        blocking = set(self._find_immediate_wins(board, attacker))
        if blocking:
            return sorted(blocking)

        return self._order_moves(board, board.get_candidate_moves(), defender, max_candidates=None)

    def _order_moves(
        self,
        board: Board,
        moves: list[Move],
        current_player: Player,
        max_candidates: Optional[int],
    ) -> list[Move]:
        """Order defensive probes by local attack+defense hotness.

        The blocking path should stay broader than the strict VCF defense set:
        after we confirm the opponent has a VCF, we probe promising ordinary
        defensive moves and verify whether each move breaks the whole line.
        """
        scored: list[tuple[int, int, float]] = []
        opponent = self._opponent_of(current_player)

        for row, col in moves:
            _, attack_score = self._analyze_move_for_player(board, row, col, current_player)
            _, defend_score = self._analyze_move_for_player(board, row, col, opponent)
            hotness = attack_score * 2 + defend_score * 3
            scored.append((row, col, hotness))

        scored.sort(key=lambda item: item[2], reverse=True)
        ordered = [(row, col) for row, col, _ in scored]
        if max_candidates is None:
            return ordered
        return ordered[:max_candidates]

    @staticmethod
    def _line_tactical_score(length: int, open_ends: int) -> int:
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
        grid = board.grid
        r, c = row + dr, col + dc
        length = 0
        while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == player:
            length += 1
            r += dr
            c += dc
        is_open = 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == Player.NONE
        return length, is_open

    def _quick_pattern_summary(
        self,
        board: Board,
        row: int,
        col: int,
        player: Player,
    ) -> tuple[bool, bool, bool]:
        if board.grid[row, col] != Player.NONE:
            return False, False, False

        has_immediate_threat = False
        has_potential = False
        promising_directions = 0
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            left_len, left_open = self._count_one_side(board, row, col, -dr, -dc, player)
            right_len, right_open = self._count_one_side(board, row, col, dr, dc, player)
            total_len = 1 + left_len + right_len
            open_ends = int(left_open) + int(right_open)

            if total_len >= 5:
                return True, False, False
            if total_len == 4 and open_ends == 2:
                return False, True, False
            if total_len == 4 and open_ends == 1:
                has_immediate_threat = True
            if total_len == 3 and open_ends >= 1:
                has_immediate_threat = True
            if total_len >= 2:
                promising_directions += 1
            if total_len >= 3 and open_ends >= 1:
                has_potential = True

        return False, False, has_immediate_threat or has_potential or promising_directions >= 2

    def _analyze_move_for_player(
        self,
        board: Board,
        row: int,
        col: int,
        player: Player,
    ) -> tuple[bool, int]:
        if board.grid[row, col] != Player.NONE:
            return False, -1

        if _analyze_moves_native is not None:
            analysis = _analyze_moves_native(board.grid, [(row, col)], int(player))[0]
            return bool(analysis[0]), int(analysis[1])

        directional_scores: list[int] = []
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
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

        center = board.grid.shape[0] // 2
        center_bias = 2 * board.grid.shape[0] - abs(row - center) - abs(col - center)
        return False, total_score + center_bias

    def _classify_strong_attacks(
        self,
        board: Board,
        attacker: Player,
    ) -> list[tuple[Move, int, int]]:
        """Return VCF-relevant attacking moves ordered by tactical urgency.

        The output shape is intentionally simple so the hot classification path
        can later be replaced by a Cython implementation without affecting the
        recursive search.
        """
        moves = self._prefilter_attack_moves(board, attacker)
        self.last_stats.classified_moves += len(moves)
        if not moves:
            return []

        classified = classify_attack_moves(board, moves, attacker)

        forcing: list[tuple[Move, int, int]] = []
        for info in classified:
            priority = self._ATTACK_PRIORITIES.get(info.attack_type, 0)
            if priority <= 0:
                continue
            forcing.append((info.move, priority, info.attack_score))

        forcing.sort(
            key=lambda item: (item[1], item[2], item[0][0], item[0][1]),
            reverse=True,
        )
        if self.last_trace is not None and "top_level_attack_classification" not in self.last_trace:
            self.last_trace["top_level_attack_classification"] = {
                "prefiltered_moves": [list(move) for move in moves],
                "classified_moves": [
                    {
                        "move": [info.move[0], info.move[1]],
                        "attack_type": info.attack_type.name,
                        "attack_score": info.attack_score,
                    }
                    for info in classified
                ],
                "strong_attacks": [
                    {
                        "move": [move[0], move[1]],
                        "priority": priority,
                        "attack_score": score,
                    }
                    for move, priority, score in forcing
                ],
            }
        return forcing

    def _prefilter_attack_moves(
        self,
        board: Board,
        attacker: Player,
    ) -> list[Move]:
        """Return attack candidates before exact VCF classification.

        Native probes may reprioritize obviously tactical moves, but they must
        not change the candidate set itself. Exact threat typing still happens
        in Python so the VCF semantics stay easy to reason about and test.
        """
        moves = board.get_candidate_moves()
        if not moves:
            return []

        if _vcf_move_probes_native is not None:
            probes = _vcf_move_probes_native(board.grid, moves, int(attacker))
        else:
            probes = self._python_vcf_move_probes(board, moves, attacker)

        hinted = self._prioritize_attack_moves(moves, probes)
        self.last_stats.prefiltered_moves += len(hinted)
        if len(hinted) == len(moves):
            return hinted

        hinted_set = set(hinted)
        remainder = [move for move in moves if move not in hinted_set]
        return hinted + remainder

    def _python_vcf_move_probes(
        self,
        board: Board,
        moves: list[Move],
        attacker: Player,
    ) -> list[tuple[bool, bool, bool, int]]:
        probes: list[tuple[bool, bool, bool, int]] = []
        for row, col in moves:
            if board.grid[row, col] != Player.NONE:
                probes.append((False, False, False, -1))
                continue

            is_win, is_open_four, has_potential = self._quick_pattern_summary(
                board, row, col, attacker
            )
            if is_win:
                probes.append((True, False, False, 200_000))
                continue
            if is_open_four:
                _is_force, score = self._analyze_move_for_player(board, row, col, attacker)
                probes.append((False, True, False, int(score)))
                continue
            if has_potential:
                _is_force, score = self._analyze_move_for_player(board, row, col, attacker)
                probes.append((False, False, True, int(score)))
                continue
            probes.append((False, False, False, 0))
        return probes

    @staticmethod
    def _prioritize_attack_moves(
        moves: list[Move],
        probes: list[tuple[bool, bool, bool, int]],
    ) -> list[Move]:
        prioritized: list[tuple[int, int, int, int]] = []
        for index, (row_col, (is_win, is_open_four, has_potential, score)) in enumerate(
            zip(moves, probes, strict=False)
        ):
            if is_win:
                priority = 5
            elif is_open_four:
                priority = 4
            elif has_potential and int(score) >= 2_000:
                priority = 1
            else:
                continue
            prioritized.append((priority, int(score), -index, index))

        prioritized.sort(reverse=True)
        return [moves[index] for _, _, _, index in prioritized]

    def _find_immediate_wins(
        self,
        board: Board,
        player: Player,
        limit: Optional[int] = None,
    ) -> list[Move]:
        moves = board.get_candidate_moves()
        if not moves:
            return []
        self.last_stats.immediate_win_checks += len(moves)

        if _analyze_moves_native is not None:
            analyses = _analyze_moves_native(board.grid, moves, int(player))
            wins: list[Move] = []
            for move, (is_win, _score) in zip(moves, analyses, strict=False):
                if not bool(is_win):
                    continue
                wins.append(move)
                if limit is not None and len(wins) >= limit:
                    return wins
            return wins

        wins: list[Move] = []
        for row, col in moves:
            board.place(row, col, player)
            try:
                if board.check_win(row, col):
                    wins.append((row, col))
                    if limit is not None and len(wins) >= limit:
                        return wins
            finally:
                board.undo()
        return wins

    @staticmethod
    def _opponent_of(player: Player) -> Player:
        return Player.WHITE if player == Player.BLACK else Player.BLACK
