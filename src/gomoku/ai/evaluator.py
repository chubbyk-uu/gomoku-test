"""Board evaluation for Gomoku AI — pattern-based shape recognition."""

from enum import IntEnum
from functools import lru_cache

from gomoku.board import Board
from gomoku.config import BOARD_SIZE, Player

# ============ 棋型枚举 ============


class Shape(IntEnum):
    """七种标准五子棋棋型。"""

    FIVE = 7
    OPEN_FOUR = 6
    HALF_FOUR = 5
    OPEN_THREE = 4
    HALF_THREE = 3
    OPEN_TWO = 2
    HALF_TWO = 1


# ============ 单型基础分 ============

SHAPE_SCORE: dict[Shape, int] = {
    Shape.FIVE: 100_000,
    Shape.OPEN_FOUR: 50_000,
    Shape.HALF_FOUR: 3_000,
    Shape.OPEN_THREE: 1_000,
    Shape.HALF_THREE: 100,
    Shape.OPEN_TWO: 100,
    Shape.HALF_TWO: 10,
}

# 防守加权：对手威胁分的乘数
DEFENSE_WEIGHT: float = 1.2

_DIRECTIONS: list[tuple[int, int]] = [(1, 0), (0, 1), (1, 1), (1, -1)]
_PLAYERS: tuple[Player, Player] = (Player.BLACK, Player.WHITE)
_SHAPES_ASC: tuple[Shape, ...] = tuple(sorted(Shape, key=int))
_ZERO_COUNTS: tuple[int, ...] = (0,) * len(_SHAPES_ASC)

# 线段中的值常量
_X = 1  # 己方
_O = 2  # 对方
_E = 0  # 空位


def _extract_line(
    grid: list[list[int]], row: int, col: int, dr: int, dc: int, player_val: int
) -> list[int]:
    """沿方向 (dr, dc) 提取以 (row, col) 为中心的 9 格线段。"""
    line: list[int] = []
    for i in range(-4, 5):
        r, c = row + dr * i, col + dc * i
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            v = grid[r][c]
            if v == player_val:
                line.append(_X)
            elif v == 0:
                line.append(_E)
            else:
                line.append(_O)
        else:
            line.append(_O)
    return line


def _extract_line_from_array(
    grid: object, row: int, col: int, dr: int, dc: int, player_val: int
) -> tuple[int, ...]:
    """从 numpy 棋盘中提取 9 格线段，避免 tolist() 的复制开销。"""
    line: list[int] = []
    for i in range(-4, 5):
        r, c = row + dr * i, col + dc * i
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            v = grid[r, c]
            if v == player_val:
                line.append(_X)
            elif v == 0:
                line.append(_E)
            else:
                line.append(_O)
        else:
            line.append(_O)
    return tuple(line)


@lru_cache(maxsize=8192)
def _match_shapes_cached(line: tuple[int, ...]) -> tuple[Shape, ...]:
    """对一条 9 格线段，识别中心点（index=4）参与的所有棋型。"""
    if line[4] != _X:
        return ()

    shapes: list[Shape] = []

    for start in range(max(0, 4 - 4), min(5, 4 + 1)):
        end = start + 5
        if end > 9:
            break
        if all(line[k] == _X for k in range(start, end)):
            shapes.append(Shape.FIVE)
            return tuple(shapes)

    for start in range(max(0, 4 - 4), min(4, 4 + 1)):
        end = start + 6
        if end > 9:
            break
        window = line[start:end]
        if window[0] == _E and window[5] == _E and all(window[k] == _X for k in range(1, 5)):
            shapes.append(Shape.OPEN_FOUR)
            return tuple(shapes)

    for start in range(max(0, 4 - 4), min(4, 4 + 1)):
        end = start + 6
        if end > 9:
            break
        window = line[start:end]
        four_mid = all(window[k] == _X for k in range(1, 5))
        if four_mid:
            if window[0] == _O and window[5] == _E:
                shapes.append(Shape.HALF_FOUR)
                return tuple(shapes)
            if window[0] == _E and window[5] == _O:
                shapes.append(Shape.HALF_FOUR)
                return tuple(shapes)

    for start in range(max(0, 4 - 4), min(5, 4 + 1)):
        end = start + 5
        if end > 9:
            break
        w = line[start:end]
        if w == (_X, _E, _X, _X, _X):
            shapes.append(Shape.HALF_FOUR)
            return tuple(shapes)
        if w == (_X, _X, _X, _E, _X):
            shapes.append(Shape.HALF_FOUR)
            return tuple(shapes)
        if w == (_X, _X, _E, _X, _X):
            shapes.append(Shape.HALF_FOUR)
            return tuple(shapes)

    for start in range(max(0, 4 - 4), min(4, 4 + 1)):
        end = start + 6
        if end > 9:
            break
        w = line[start:end]
        if w == (_E, _E, _X, _X, _X, _E):
            shapes.append(Shape.OPEN_THREE)
            return tuple(shapes)
        if w == (_E, _X, _X, _X, _E, _E):
            shapes.append(Shape.OPEN_THREE)
            return tuple(shapes)

    for start in range(max(0, 4 - 5), min(4, 4 + 1)):
        end = start + 6
        if end > 9:
            break
        w = line[start:end]
        if w == (_E, _X, _E, _X, _X, _E):
            shapes.append(Shape.OPEN_THREE)
            return tuple(shapes)
        if w == (_E, _X, _X, _E, _X, _E):
            shapes.append(Shape.OPEN_THREE)
            return tuple(shapes)

    for start in range(max(0, 4 - 5), min(3, 4 + 1)):
        end = start + 7
        if end > 9:
            break
        w = line[start:end]
        if w == (_O, _E, _X, _X, _X, _E, _E):
            shapes.append(Shape.HALF_THREE)
            return tuple(shapes)
        if w == (_E, _E, _X, _X, _X, _E, _O):
            shapes.append(Shape.HALF_THREE)
            return tuple(shapes)

    for start in range(max(0, 4 - 4), min(4, 4 + 1)):
        end = start + 6
        if end > 9:
            break
        w = line[start:end]
        if w == (_O, _X, _X, _X, _E, _E):
            shapes.append(Shape.HALF_THREE)
            return tuple(shapes)
        if w == (_E, _E, _X, _X, _X, _O):
            shapes.append(Shape.HALF_THREE)
            return tuple(shapes)

    for start in range(max(0, 4 - 5), min(4, 4 + 1)):
        end = start + 6
        if end > 9:
            break
        w = line[start:end]
        if w == (_O, _X, _E, _X, _X, _E):
            shapes.append(Shape.HALF_THREE)
            return tuple(shapes)
        if w == (_E, _X, _X, _E, _X, _O):
            shapes.append(Shape.HALF_THREE)
            return tuple(shapes)
        if w == (_O, _X, _X, _E, _X, _E):
            shapes.append(Shape.HALF_THREE)
            return tuple(shapes)
        if w == (_E, _X, _E, _X, _X, _O):
            shapes.append(Shape.HALF_THREE)
            return tuple(shapes)

    for start in range(max(0, 4 - 4), min(4, 4 + 1)):
        end = start + 6
        if end > 9:
            break
        w = line[start:end]
        if w == (_E, _E, _X, _X, _E, _E):
            shapes.append(Shape.OPEN_TWO)
            return tuple(shapes)

    for start in range(max(0, 4 - 4), min(5, 4 + 1)):
        end = start + 5
        if end > 9:
            break
        w = line[start:end]
        if w == (_E, _X, _E, _X, _E):
            shapes.append(Shape.OPEN_TWO)
            return tuple(shapes)

    for start in range(max(0, 4 - 4), min(4, 4 + 1)):
        end = start + 6
        if end > 9:
            break
        w = line[start:end]
        if w == (_O, _X, _X, _E, _E, _E):
            shapes.append(Shape.HALF_TWO)
            return tuple(shapes)
        if w == (_E, _E, _E, _X, _X, _O):
            shapes.append(Shape.HALF_TWO)
            return tuple(shapes)

    for start in range(max(0, 4 - 5), min(4, 4 + 1)):
        end = start + 6
        if end > 9:
            break
        w = line[start:end]
        if w == (_O, _X, _E, _X, _E, _E):
            shapes.append(Shape.HALF_TWO)
            return tuple(shapes)
        if w == (_E, _E, _X, _E, _X, _O):
            shapes.append(Shape.HALF_TWO)
            return tuple(shapes)

    return tuple(shapes)


def _match_shapes(line: list[int]) -> list[Shape]:
    """列表版本包装，供测试和外部调用。"""
    return list(_match_shapes_cached(tuple(line)))


def _coords_for_line(direction_index: int, line_id: int) -> list[tuple[int, int]]:
    """返回某个方向下的整条线坐标，顺序与旧算法扫描顺序一致。"""
    if direction_index == 0:
        return [(row, line_id) for row in range(BOARD_SIZE)]
    if direction_index == 1:
        return [(line_id, col) for col in range(BOARD_SIZE)]
    if direction_index == 2:
        k = line_id - (BOARD_SIZE - 1)
        row = max(0, -k)
        col = row + k
        coords: list[tuple[int, int]] = []
        while row < BOARD_SIZE and col < BOARD_SIZE:
            coords.append((row, col))
            row += 1
            col += 1
        return coords

    total = line_id
    row = max(0, total - (BOARD_SIZE - 1))
    coords = []
    while row < BOARD_SIZE:
        col = total - row
        if 0 <= col < BOARD_SIZE:
            coords.append((row, col))
        row += 1
    return coords


def _line_id_for_cell(direction_index: int, row: int, col: int) -> int:
    if direction_index == 0:
        return col
    if direction_index == 1:
        return row
    if direction_index == 2:
        return col - row + BOARD_SIZE - 1
    return row + col


@lru_cache(maxsize=128)
def _line_coords(direction_index: int, line_id: int) -> tuple[tuple[int, int], ...]:
    return tuple(_coords_for_line(direction_index, line_id))


def _counts_to_tuple(counts: dict[Shape, int]) -> tuple[int, ...]:
    return tuple(counts[shape] for shape in _SHAPES_ASC)


def _tuple_to_counts(values: tuple[int, ...]) -> dict[Shape, int]:
    return {shape: values[idx] for idx, shape in enumerate(_SHAPES_ASC)}


def _tuple_add(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(x + y for x, y in zip(a, b))


def _tuple_sub(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(x - y for x, y in zip(a, b))


def _count_shapes_on_line(
    board: Board,
    player: Player,
    direction_index: int,
    line_id: int,
) -> tuple[int, ...]:
    """按单条线复现旧版 seen 语义，得到精确棋型计数。"""
    coords = _line_coords(direction_index, line_id)
    player_val = int(player)
    seen_indices: set[int] = set()
    counts = {shape: 0 for shape in Shape}

    dr, dc = _DIRECTIONS[direction_index]
    for idx, (row, col) in enumerate(coords):
        if idx in seen_indices or board.grid[row, col] != player_val:
            continue
        matched = _match_shapes_cached(
            _extract_line_from_array(board.grid, row, col, dr, dc, player_val)
        )
        for shape in matched:
            counts[shape] += 1
            start = max(0, idx - 4)
            end = min(len(coords), idx + 5)
            for seen_idx in range(start, end):
                seen_row, seen_col = coords[seen_idx]
                if board.grid[seen_row, seen_col] == player_val:
                    seen_indices.add(seen_idx)

    return _counts_to_tuple(counts)


def _count_shapes_legacy(board: Board, player: Player) -> dict[Shape, int]:
    """旧版全盘扫描实现，保留用于等价性校验。"""
    grid = board.grid
    player_val = int(player)
    counts: dict[Shape, int] = {s: 0 for s in Shape}
    seen: set[tuple[int, int, int, int]] = set()

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if grid[i][j] != player_val:
                continue
            for dr, dc in _DIRECTIONS:
                key = (i, j, dr, dc)
                if key in seen:
                    continue
                line = _extract_line_from_array(grid, i, j, dr, dc, player_val)
                matched = _match_shapes_cached(line)
                for shape in matched:
                    counts[shape] += 1
                    for step in range(-4, 5):
                        r, c = i + dr * step, j + dc * step
                        if (
                            0 <= r < BOARD_SIZE
                            and 0 <= c < BOARD_SIZE
                            and grid[r, c] == player_val
                        ):
                            seen.add((r, c, dr, dc))

    return counts


class _BoardEvalState:
    """维护按线分解的精确棋型计数，支持落子/悔棋增量更新。"""

    def __init__(self, board: Board) -> None:
        self.line_counts: dict[Player, dict[tuple[int, int], tuple[int, ...]]] = {
            player: {} for player in _PLAYERS
        }
        self.total_counts: dict[Player, tuple[int, ...]] = {
            player: _ZERO_COUNTS for player in _PLAYERS
        }
        self._initialize(board)

    def _initialize(self, board: Board) -> None:
        for player in _PLAYERS:
            total = _ZERO_COUNTS
            for direction_index in range(len(_DIRECTIONS)):
                max_line_id = BOARD_SIZE if direction_index < 2 else BOARD_SIZE * 2 - 1
                for line_id in range(max_line_id):
                    counts = _count_shapes_on_line(board, player, direction_index, line_id)
                    self.line_counts[player][(direction_index, line_id)] = counts
                    total = _tuple_add(total, counts)
            self.total_counts[player] = total

    def on_board_changed(self, board: Board, row: int, col: int) -> None:
        for player in _PLAYERS:
            total = self.total_counts[player]
            for direction_index in range(len(_DIRECTIONS)):
                line_id = _line_id_for_cell(direction_index, row, col)
                key = (direction_index, line_id)
                old_counts = self.line_counts[player][key]
                new_counts = _count_shapes_on_line(board, player, direction_index, line_id)
                if new_counts != old_counts:
                    total = _tuple_sub(total, old_counts)
                    total = _tuple_add(total, new_counts)
                    self.line_counts[player][key] = new_counts
            self.total_counts[player] = total


def _ensure_eval_state(board: Board) -> _BoardEvalState:
    state = board._eval_state
    if state is None:
        state = _BoardEvalState(board)
        board._eval_state = state
    return state


def _count_shapes(board: Board, player: Player) -> dict[Shape, int]:
    """统计一方在棋盘上的所有棋型数量。"""
    state = _ensure_eval_state(board)
    return _tuple_to_counts(state.total_counts[player])


def _calc_total(counts: dict[Shape, int]) -> int:
    """根据棋型数量计算总分（含组合加成）。"""
    if counts[Shape.FIVE] > 0:
        return SHAPE_SCORE[Shape.FIVE]

    open_fours = counts[Shape.OPEN_FOUR]
    half_fours = counts[Shape.HALF_FOUR]
    open_threes = counts[Shape.OPEN_THREE]
    half_threes = counts[Shape.HALF_THREE]

    if open_fours >= 1:
        return 50_000
    if half_fours >= 2:
        return 50_000
    if half_fours >= 1 and open_threes >= 1:
        return 10_000
    if open_threes >= 2:
        return 10_000
    if half_fours >= 1 and half_threes >= 1:
        return 5_000
    if open_threes >= 1 and half_threes >= 1:
        return 3_000

    total = 0
    for shape, count in counts.items():
        total += SHAPE_SCORE[shape] * count
    return total


def evaluate(board: Board, ai_player: Player) -> int:
    """评估当前棋盘对 ai_player 的净分值。"""
    opponent = Player.WHITE if ai_player == Player.BLACK else Player.BLACK
    ai_counts = _count_shapes(board, ai_player)
    opp_counts = _count_shapes(board, opponent)
    ai_score = _calc_total(ai_counts)
    opp_score = _calc_total(opp_counts)
    return ai_score - int(opp_score * DEFENSE_WEIGHT)
