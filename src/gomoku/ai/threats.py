"""Threat classification utilities for Gomoku search."""

from dataclasses import dataclass
from enum import IntEnum

from gomoku.ai.evaluator import Shape, _count_shapes
from gomoku.board import Board
from gomoku.config import Player

_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (1, -1))


class ThreatType(IntEnum):
    """Threat categories aligned with the evaluator's pattern language."""

    OTHER = 0
    OPEN_THREE = 1
    HALF_FOUR = 2
    DOUBLE_OPEN_THREE = 3
    FOUR_THREE = 4
    DOUBLE_HALF_FOUR = 5
    OPEN_FOUR = 6
    BLOCK_WIN = 7
    WIN = 8


@dataclass(frozen=True)
class ThreatInfo:
    """Threat analysis result for a single candidate move."""

    move: tuple[int, int]
    threat_type: ThreatType
    attack_type: ThreatType
    defense_type: ThreatType
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


def _quick_pattern_summary(
    board: Board,
    row: int,
    col: int,
    player: Player,
) -> tuple[bool, bool, bool]:
    """Cheap local prefilter for strong tactical patterns around one move."""
    if board.grid[row, col] != Player.NONE:
        return False, False, False

    has_immediate_threat = False
    has_potential = False
    promising_directions = 0
    for dr, dc in _DIRECTIONS:
        left_len, left_open = _count_one_side(board, row, col, -dr, -dc, player)
        right_len, right_open = _count_one_side(board, row, col, dr, dc, player)
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


def _threat_score(threat_type: ThreatType) -> int:
    return {
        ThreatType.WIN: 200_000,
        ThreatType.OPEN_FOUR: 120_000,
        ThreatType.DOUBLE_HALF_FOUR: 90_000,
        ThreatType.FOUR_THREE: 70_000,
        ThreatType.DOUBLE_OPEN_THREE: 55_000,
        ThreatType.HALF_FOUR: 30_000,
        ThreatType.OPEN_THREE: 12_000,
        ThreatType.OTHER: 0,
        ThreatType.BLOCK_WIN: 180_000,
    }[threat_type]


def _classify_from_counts(counts: dict[Shape, int]) -> ThreatType:
    """Map exact evaluator counts to a forcing-search threat category."""
    if counts[Shape.FIVE] > 0:
        return ThreatType.WIN
    if counts[Shape.OPEN_FOUR] > 0:
        return ThreatType.OPEN_FOUR
    if counts[Shape.HALF_FOUR] >= 2:
        return ThreatType.DOUBLE_HALF_FOUR
    if counts[Shape.HALF_FOUR] >= 1 and counts[Shape.OPEN_THREE] >= 1:
        return ThreatType.FOUR_THREE
    if counts[Shape.OPEN_THREE] >= 2:
        return ThreatType.DOUBLE_OPEN_THREE
    if counts[Shape.HALF_FOUR] >= 1:
        return ThreatType.HALF_FOUR
    if counts[Shape.OPEN_THREE] >= 1:
        return ThreatType.OPEN_THREE
    return ThreatType.OTHER


def _classify_move_for_player(board: Board, row: int, col: int, player: Player) -> ThreatType:
    if board.grid[row, col] != Player.NONE:
        return ThreatType.OTHER
    is_win, is_open_four, has_potential = _quick_pattern_summary(board, row, col, player)
    if is_win:
        return ThreatType.WIN
    if is_open_four:
        return ThreatType.OPEN_FOUR
    if not has_potential:
        return ThreatType.OTHER

    board.place(row, col, player)
    try:
        counts = _count_shapes(board, player)
    finally:
        board.undo()
    return _classify_from_counts(counts)


def classify_move(board: Board, row: int, col: int, player: Player) -> ThreatInfo:
    """Classify a candidate move using exact evaluator pattern counts."""
    opponent = Player.WHITE if player == Player.BLACK else Player.BLACK

    attack_type = _classify_move_for_player(board, row, col, player)
    defense_type = _classify_move_for_player(board, row, col, opponent)

    if attack_type == ThreatType.WIN:
        threat_type = ThreatType.WIN
    elif defense_type == ThreatType.WIN:
        threat_type = ThreatType.BLOCK_WIN
    else:
        threat_type = attack_type

    return ThreatInfo(
        move=(row, col),
        threat_type=threat_type,
        attack_type=attack_type,
        defense_type=defense_type,
        attack_score=_threat_score(attack_type),
        defense_score=_threat_score(
            ThreatType.BLOCK_WIN if defense_type == ThreatType.WIN else defense_type
        ),
    )


def classify_moves(
    board: Board,
    moves: list[tuple[int, int]],
    player: Player,
) -> list[ThreatInfo]:
    """Classify a list of candidate moves for the given player."""
    return [classify_move(board, row, col, player) for row, col in moves]


def classify_attack_moves(
    board: Board,
    moves: list[tuple[int, int]],
    player: Player,
) -> list[ThreatInfo]:
    """Classify only attacking threats for the given player."""
    infos: list[ThreatInfo] = []
    for row, col in moves:
        attack_type = _classify_move_for_player(board, row, col, player)
        infos.append(
            ThreatInfo(
                move=(row, col),
                threat_type=attack_type,
                attack_type=attack_type,
                defense_type=ThreatType.OTHER,
                attack_score=_threat_score(attack_type),
                defense_score=0,
            )
        )
    return infos


def classify_defense_moves(
    board: Board,
    moves: list[tuple[int, int]],
    player: Player,
) -> list[ThreatInfo]:
    """Classify only defensive threats against the given player."""
    opponent = Player.WHITE if player == Player.BLACK else Player.BLACK
    infos: list[ThreatInfo] = []
    for row, col in moves:
        defense_type = _classify_move_for_player(board, row, col, opponent)
        threat_type = ThreatType.BLOCK_WIN if defense_type == ThreatType.WIN else defense_type
        infos.append(
            ThreatInfo(
                move=(row, col),
                threat_type=threat_type,
                attack_type=ThreatType.OTHER,
                defense_type=defense_type,
                attack_score=0,
                defense_score=_threat_score(threat_type),
            )
        )
    return infos
