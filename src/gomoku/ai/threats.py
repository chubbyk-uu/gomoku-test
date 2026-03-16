"""Threat classification utilities for Gomoku search."""

from dataclasses import dataclass
from enum import IntEnum

from gomoku.board import Board
from gomoku.config import Player

_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (1, -1))


class ThreatType(IntEnum):
    """Coarse threat categories used to guide forcing search."""

    OTHER = 0
    FOUR_THREE = 1
    OPEN_FOUR = 2
    BLOCK_WIN = 3
    WIN = 4


@dataclass(frozen=True)
class ThreatInfo:
    """Threat analysis result for a single candidate move."""

    move: tuple[int, int]
    threat_type: ThreatType
    attack_score: int
    defense_score: int


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


def _analyze_patterns(
    board: Board,
    row: int,
    col: int,
    player: Player,
) -> tuple[bool, bool, bool, int]:
    """Return immediate-win/open-four/open-three flags plus a coarse attack score."""
    if board.grid[row, col] != Player.NONE:
        return False, False, False, -1

    has_open_four = False
    open_three_count = 0
    attack_score = 0

    for dr, dc in _DIRECTIONS:
        left_len, left_open = _count_one_side(board, row, col, -dr, -dc, player)
        right_len, right_open = _count_one_side(board, row, col, dr, dc, player)
        total_len = 1 + left_len + right_len
        open_ends = int(left_open) + int(right_open)

        if total_len >= 5:
            return True, False, False, 200_000
        if total_len == 4 and open_ends == 2:
            has_open_four = True
            attack_score += 80_000
        elif total_len == 4 and open_ends == 1:
            attack_score += 30_000
        elif total_len == 3 and open_ends == 2:
            open_three_count += 1
            attack_score += 8_000
        elif total_len == 3 and open_ends == 1:
            attack_score += 2_000
        elif total_len == 2 and open_ends == 2:
            attack_score += 500
        elif total_len == 2 and open_ends == 1:
            attack_score += 100

    return False, has_open_four, open_three_count >= 1, attack_score


def classify_move(board: Board, row: int, col: int, player: Player) -> ThreatInfo:
    """Classify a candidate move into a coarse threat category."""
    opponent = Player.WHITE if player == Player.BLACK else Player.BLACK

    is_win, has_open_four, has_open_three, attack_score = _analyze_patterns(
        board, row, col, player
    )
    opp_is_win, _, _, defense_score = _analyze_patterns(board, row, col, opponent)

    if is_win:
        threat_type = ThreatType.WIN
    elif opp_is_win:
        threat_type = ThreatType.BLOCK_WIN
    elif has_open_four:
        threat_type = ThreatType.OPEN_FOUR
    elif has_open_three and attack_score >= 10_000:
        threat_type = ThreatType.FOUR_THREE
    else:
        threat_type = ThreatType.OTHER

    return ThreatInfo(
        move=(row, col),
        threat_type=threat_type,
        attack_score=attack_score,
        defense_score=defense_score,
    )


def classify_moves(
    board: Board,
    moves: list[tuple[int, int]],
    player: Player,
) -> list[ThreatInfo]:
    """Classify a list of candidate moves for the given player."""
    return [classify_move(board, row, col, player) for row, col in moves]
