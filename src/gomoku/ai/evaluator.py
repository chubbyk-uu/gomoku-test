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

# 线段中的值常量
_X = 1  # 己方
_O = 2  # 对方
_E = 0  # 空位


def _extract_line(
    grid: list[list[int]], row: int, col: int, dr: int, dc: int, player_val: int
) -> list[int]:
    """沿方向 (dr, dc) 提取以 (row, col) 为中心的 9 格线段。

    返回的线段中：1=己方, 2=对方, 0=空位。
    越界位置视为对方棋子 (2)。

    Args:
        grid: 棋盘二维列表。
        row, col: 中心棋子坐标。
        dr, dc: 方向向量。
        player_val: 己方棋子的整数值。

    Returns:
        长度为 9 的列表。
    """
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
            line.append(_O)  # 越界视为对方（封堵）
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
    """对一条 9 格线段，识别中心点（index=4）参与的所有棋型。

    中心点必须是己方棋子 (_X)。

    Args:
        line: 长度为 9 的线段，值为 _X/_O/_E。

    Returns:
        识别到的棋型列表（可能为空或含多个）。
    """
    if line[4] != _X:
        return ()

    shapes: list[Shape] = []

    # 在线段中搜索所有包含 index=4 的窗口进行模式匹配
    # FIVE: 连续 5 个 X，窗口大小 5
    for start in range(max(0, 4 - 4), min(5, 4 + 1)):
        end = start + 5
        if end > 9:
            break
        if all(line[k] == _X for k in range(start, end)):
            shapes.append(Shape.FIVE)
            return tuple(shapes)  # 连五直接返回，无需继续

    # OPEN_FOUR: _XXXX_  窗口大小 6
    for start in range(max(0, 4 - 4), min(4, 4 + 1)):
        end = start + 6
        if end > 9:
            break
        window = line[start:end]
        if (window[0] == _E and window[5] == _E
                and all(window[k] == _X for k in range(1, 5))):
            shapes.append(Shape.OPEN_FOUR)
            return tuple(shapes)  # 活四直接返回

    # HALF_FOUR: 冲四（含跳冲四）
    # 连冲四: OXXXX_ 或 _XXXXO  窗口大小 6
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

    # 跳冲四: X_XXX (5格), XXX_X (5格), XX_XX (5格)
    # 这些模式中有4个X和1个空位，且两端至少一端被封堵或只看内部
    # X_XXX: 窗口 5
    for start in range(max(0, 4 - 4), min(5, 4 + 1)):
        end = start + 5
        if end > 9:
            break
        w = line[start:end]
        if (w[0] == _X and w[1] == _E and w[2] == _X and w[3] == _X and w[4] == _X):
            shapes.append(Shape.HALF_FOUR)
            return tuple(shapes)
        if (w[0] == _X and w[1] == _X and w[2] == _X and w[3] == _E and w[4] == _X):
            shapes.append(Shape.HALF_FOUR)
            return tuple(shapes)
        if (w[0] == _X and w[1] == _X and w[2] == _E and w[3] == _X and w[4] == _X):
            shapes.append(Shape.HALF_FOUR)
            return tuple(shapes)

    # OPEN_THREE: 活三（含跳活三）
    # 连活三: __XXX_ (6格) 或 _XXX__ (6格)
    # 由于对称性，只需检查 _?XXX?_ 形式
    for start in range(max(0, 4 - 4), min(4, 4 + 1)):
        end = start + 6
        if end > 9:
            break
        w = line[start:end]
        # __XXX_
        if (w[0] == _E and w[1] == _E and w[2] == _X and w[3] == _X
                and w[4] == _X and w[5] == _E):
            shapes.append(Shape.OPEN_THREE)
            return tuple(shapes)
        # _XXX__
        if (w[0] == _E and w[1] == _X and w[2] == _X and w[3] == _X
                and w[4] == _E and w[5] == _E):
            shapes.append(Shape.OPEN_THREE)
            return tuple(shapes)

    # 跳活三: _X_XX_ (6格), _XX_X_ (6格)
    for start in range(max(0, 4 - 5), min(4, 4 + 1)):
        end = start + 6
        if end > 9:
            break
        w = line[start:end]
        if (w[0] == _E and w[1] == _X and w[2] == _E and w[3] == _X
                and w[4] == _X and w[5] == _E):
            shapes.append(Shape.OPEN_THREE)
            return tuple(shapes)
        if (w[0] == _E and w[1] == _X and w[2] == _X and w[3] == _E
                and w[4] == _X and w[5] == _E):
            shapes.append(Shape.OPEN_THREE)
            return tuple(shapes)

    # HALF_THREE: 眠三（含跳眠三）
    # 连眠三: O_XXX__ (7格), __XXX_O (7格)
    for start in range(max(0, 4 - 5), min(3, 4 + 1)):
        end = start + 7
        if end > 9:
            break
        w = line[start:end]
        # O_XXX__
        if (w[0] == _O and w[1] == _E and w[2] == _X and w[3] == _X
                and w[4] == _X and w[5] == _E and w[6] == _E):
            shapes.append(Shape.HALF_THREE)
            return tuple(shapes)
        # __XXX_O
        if (w[0] == _E and w[1] == _E and w[2] == _X and w[3] == _X
                and w[4] == _X and w[5] == _E and w[6] == _O):
            shapes.append(Shape.HALF_THREE)
            return tuple(shapes)

    # 连眠三 (6格): OXXX__ 或 __XXXO
    for start in range(max(0, 4 - 4), min(4, 4 + 1)):
        end = start + 6
        if end > 9:
            break
        w = line[start:end]
        # OXXX__
        if (w[0] == _O and w[1] == _X and w[2] == _X and w[3] == _X
                and w[4] == _E and w[5] == _E):
            shapes.append(Shape.HALF_THREE)
            return tuple(shapes)
        # __XXXO
        if (w[0] == _E and w[1] == _E and w[2] == _X and w[3] == _X
                and w[4] == _X and w[5] == _O):
            shapes.append(Shape.HALF_THREE)
            return tuple(shapes)

    # 跳眠三: OX_XX_ (6格), _XX_XO (6格), OXX_X_ (6格), _X_XXO (6格)
    for start in range(max(0, 4 - 5), min(4, 4 + 1)):
        end = start + 6
        if end > 9:
            break
        w = line[start:end]
        if (w[0] == _O and w[1] == _X and w[2] == _E and w[3] == _X
                and w[4] == _X and w[5] == _E):
            shapes.append(Shape.HALF_THREE)
            return tuple(shapes)
        if (w[0] == _E and w[1] == _X and w[2] == _X and w[3] == _E
                and w[4] == _X and w[5] == _O):
            shapes.append(Shape.HALF_THREE)
            return tuple(shapes)
        if (w[0] == _O and w[1] == _X and w[2] == _X and w[3] == _E
                and w[4] == _X and w[5] == _E):
            shapes.append(Shape.HALF_THREE)
            return tuple(shapes)
        if (w[0] == _E and w[1] == _X and w[2] == _E and w[3] == _X
                and w[4] == _X and w[5] == _O):
            shapes.append(Shape.HALF_THREE)
            return tuple(shapes)

    # OPEN_TWO: 活二（含跳活二）
    # 连活二: __XX__ (6格)
    for start in range(max(0, 4 - 4), min(4, 4 + 1)):
        end = start + 6
        if end > 9:
            break
        w = line[start:end]
        if (w[0] == _E and w[1] == _E and w[2] == _X and w[3] == _X
                and w[4] == _E and w[5] == _E):
            shapes.append(Shape.OPEN_TWO)
            return tuple(shapes)

    # 跳活二: _X_X_ (5格)
    for start in range(max(0, 4 - 4), min(5, 4 + 1)):
        end = start + 5
        if end > 9:
            break
        w = line[start:end]
        if (w[0] == _E and w[1] == _X and w[2] == _E and w[3] == _X and w[4] == _E):
            shapes.append(Shape.OPEN_TWO)
            return tuple(shapes)

    # HALF_TWO: 眠二
    # 连眠二: OXX___ (6格), ___XXO (6格)
    for start in range(max(0, 4 - 4), min(4, 4 + 1)):
        end = start + 6
        if end > 9:
            break
        w = line[start:end]
        if (w[0] == _O and w[1] == _X and w[2] == _X and w[3] == _E
                and w[4] == _E and w[5] == _E):
            shapes.append(Shape.HALF_TWO)
            return tuple(shapes)
        if (w[0] == _E and w[1] == _E and w[2] == _E and w[3] == _X
                and w[4] == _X and w[5] == _O):
            shapes.append(Shape.HALF_TWO)
            return tuple(shapes)

    # 跳眠二: OX_X__ (6格), __X_XO (6格)
    for start in range(max(0, 4 - 5), min(4, 4 + 1)):
        end = start + 6
        if end > 9:
            break
        w = line[start:end]
        if (w[0] == _O and w[1] == _X and w[2] == _E and w[3] == _X
                and w[4] == _E and w[5] == _E):
            shapes.append(Shape.HALF_TWO)
            return tuple(shapes)
        if (w[0] == _E and w[1] == _E and w[2] == _X and w[3] == _E
                and w[4] == _X and w[5] == _O):
            shapes.append(Shape.HALF_TWO)
            return tuple(shapes)

    return tuple(shapes)


def _match_shapes(line: list[int]) -> list[Shape]:
    """列表版本包装，供测试和外部调用。"""
    return list(_match_shapes_cached(tuple(line)))


def _count_shapes(board: Board, player: Player) -> dict[Shape, int]:
    """统计一方在棋盘上的所有棋型数量。

    为避免同一条线上重复计数，对每个棋子沿每个方向只在该棋子是该方向上
    「最靠近起始方向」的己方棋子时才计数（即沿反方向的相邻格不是己方棋子）。

    Args:
        board: 当前棋盘状态。
        player: 待评估的一方。

    Returns:
        各棋型到数量的映射。
    """
    grid = board.grid
    player_val = int(player)
    counts: dict[Shape, int] = {s: 0 for s in Shape}
    # 记录已识别的 (棋子位置, 方向) 组合，避免同一条棋型被多个棋子重复计数
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
                    # 标记该方向上其他己方棋子，避免重复
                    for step in range(-4, 5):
                        r, c = i + dr * step, j + dc * step
                        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                            if grid[r, c] == player_val:
                                seen.add((r, c, dr, dc))

    return counts


def _calc_total(counts: dict[Shape, int]) -> int:
    """根据棋型数量计算总分（含组合加成）。

    组合加成采用替代逻辑：当检测到强组合时，用组合分替代基础分累加。

    Args:
        counts: 各棋型的数量。

    Returns:
        总分值。
    """
    # 先检查是否有连五
    if counts[Shape.FIVE] > 0:
        return SHAPE_SCORE[Shape.FIVE]

    # 检查组合加成（从高到低优先级）
    open_fours = counts[Shape.OPEN_FOUR]
    half_fours = counts[Shape.HALF_FOUR]
    open_threes = counts[Shape.OPEN_THREE]
    half_threes = counts[Shape.HALF_THREE]

    # 有活四 → 必胜
    if open_fours >= 1:
        return 50_000

    # 双冲四 → 必胜
    if half_fours >= 2:
        return 50_000

    # 冲四 + 活三 → 近必胜
    if half_fours >= 1 and open_threes >= 1:
        return 10_000

    # 双活三 → 近必胜
    if open_threes >= 2:
        return 10_000

    # 冲四 + 眠三
    if half_fours >= 1 and half_threes >= 1:
        return 5_000

    # 活三 + 眠三
    if open_threes >= 1 and half_threes >= 1:
        return 3_000

    # 无组合：累加基础分
    total = 0
    for shape, count in counts.items():
        total += SHAPE_SCORE[shape] * count
    return total


def evaluate(board: Board, ai_player: Player) -> int:
    """评估当前棋盘对 ai_player 的净分值。

    Args:
        board: 当前棋盘状态。
        ai_player: AI 执棋颜色。

    Returns:
        AI 总分 − 对手加权总分（正值对 AI 有利）。
    """
    opponent = Player.WHITE if ai_player == Player.BLACK else Player.BLACK
    ai_counts = _count_shapes(board, ai_player)
    opp_counts = _count_shapes(board, opponent)
    ai_score = _calc_total(ai_counts)
    opp_score = _calc_total(opp_counts)
    return ai_score - int(opp_score * DEFENSE_WEIGHT)
