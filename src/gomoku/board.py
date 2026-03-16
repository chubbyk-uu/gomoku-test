"""Gomoku board logic."""

import random as _random
from typing import Optional

import numpy as np

from gomoku.config import BOARD_SIZE, Player

_CANDIDATE_RANGE = 2  # 候选点搜索半径

# Zobrist 哈希表：_ZOBRIST[row][col][player_index]，player_index: BLACK=0, WHITE=1
# 固定随机种子保证每次进程内一致
_rng = _random.Random(42)
_ZOBRIST: list[list[list[int]]] = [
    [[_rng.getrandbits(64), _rng.getrandbits(64)] for _ in range(BOARD_SIZE)]
    for _ in range(BOARD_SIZE)
]


class Board:
    """封装五子棋棋盘状态与操作。

    Attributes:
        grid: 15x15 numpy 数组（int8），值为 Player 枚举的整数值。
        move_history: 落子历史，每条记录为 (row, col, player)。
        last_move: 最后一手坐标 (row, col)，棋盘为空时为 None。
        hash: 当前局面的 Zobrist 哈希值（64 位整数），随落子/悔棋增量更新。
    """

    def __init__(self) -> None:
        self.grid: np.ndarray = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.move_history: list[tuple[int, int, Player]] = []
        self.last_move: Optional[tuple[int, int]] = None
        self.hash: int = 0
        # 增量候选点集合，及每步的增删记录（用于 undo 精确恢复）
        self._candidates: set[tuple[int, int]] = set()
        self._candidate_history: list[tuple[set[tuple[int, int]], set[tuple[int, int]]]] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def place(self, row: int, col: int, player: Player) -> bool:
        """在 (row, col) 落子。

        Args:
            row: 行坐标 [0, BOARD_SIZE)。
            col: 列坐标 [0, BOARD_SIZE)。
            player: 落子方，必须是 Player.BLACK 或 Player.WHITE。

        Returns:
            True 表示落子成功；False 表示坐标越界或该位置已有棋子。
        """
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return False
        if self.grid[row, col] != Player.NONE:
            return False

        # 计算候选点增量：落子前统计，确保 grid 尚未更新
        removed: set[tuple[int, int]] = set()
        if (row, col) in self._candidates:
            removed.add((row, col))

        added: set[tuple[int, int]] = set()
        for dr in range(-_CANDIDATE_RANGE, _CANDIDATE_RANGE + 1):
            for dc in range(-_CANDIDATE_RANGE, _CANDIDATE_RANGE + 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if (
                    0 <= nr < BOARD_SIZE
                    and 0 <= nc < BOARD_SIZE
                    and self.grid[nr, nc] == Player.NONE
                    and (nr, nc) not in self._candidates
                ):
                    added.add((nr, nc))

        self._candidates -= removed
        self._candidates |= added
        self._candidate_history.append((added, removed))

        self.grid[row, col] = player
        self.hash ^= _ZOBRIST[row][col][int(player) - 1]
        self.move_history.append((row, col, player))
        self.last_move = (row, col)
        return True

    def undo(self) -> Optional[tuple[int, int, Player]]:
        """撤销最后一手落子。

        Returns:
            被撤销的 (row, col, player)；历史为空时返回 None。
        """
        if not self.move_history:
            return None
        row, col, player = self.move_history.pop()
        self.grid[row, col] = Player.NONE
        self.hash ^= _ZOBRIST[row][col][int(player) - 1]
        self.last_move = (
            (self.move_history[-1][0], self.move_history[-1][1]) if self.move_history else None
        )

        # 精确恢复候选点集合
        added, removed = self._candidate_history.pop()
        self._candidates -= added
        self._candidates |= removed

        return row, col, player

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def check_win(self, row: int, col: int) -> bool:
        """检查 (row, col) 处的棋子是否构成五连珠。

        通过四个方向的双向扩展，统计该棋子所在连续段长度。

        Args:
            row: 行坐标。
            col: 列坐标。

        Returns:
            True 表示该位置构成胜利；空位返回 False。
        """
        player_val = int(self.grid[row, col])
        if player_val == Player.NONE:
            return False

        def _count_direction(dr: int, dc: int) -> int:
            length = 1

            r, c = row + dr, col + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.grid[r, c] == player_val:
                length += 1
                r += dr
                c += dc

            r, c = row - dr, col - dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.grid[r, c] == player_val:
                length += 1
                r -= dr
                c -= dc

            return length

        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if _count_direction(dr, dc) >= 5:
                return True
        return False

    def get_candidate_moves(self) -> list[tuple[int, int]]:
        """返回所有邻近已有棋子的空位（候选落子点）。

        若棋盘为空，直接返回天元（中心点）。候选点通过增量集合维护，O(1) 获取。

        Returns:
            候选坐标列表 [(row, col), ...]。
        """
        if not self.move_history:
            return [(BOARD_SIZE // 2, BOARD_SIZE // 2)]
        return list(self._candidates)

    def is_full(self) -> bool:
        """棋盘是否已落满。

        Returns:
            True 表示无空位。
        """
        return bool(np.all(self.grid != Player.NONE))

    def copy(self) -> "Board":
        """返回棋盘的深拷贝，用于 AI 搜索时的模拟落子。

        Returns:
            新的 Board 实例，状态与当前相同。
        """
        new_board = Board()
        new_board.grid = np.copy(self.grid)
        new_board.move_history = self.move_history.copy()
        new_board.last_move = self.last_move
        new_board.hash = self.hash
        new_board._candidates = self._candidates.copy()
        new_board._candidate_history = [(a.copy(), r.copy()) for a, r in self._candidate_history]
        return new_board
