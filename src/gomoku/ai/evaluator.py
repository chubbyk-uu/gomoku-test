"""Board evaluation for Gomoku AI."""

from gomoku.board import Board
from gomoku.config import BOARD_SIZE, Player

# 评分表：(连子数, 封堵端数) -> 分值
# blocks=0 表示两端均开放（活棋），blocks=1 表示一端被堵（冲棋），blocks=2 已无意义（跳过）
SCORE_TABLE: dict[tuple[int, int], int] = {
    (5, 0): 100_000,  # 五连（胜利）
    (5, 1): 100_000,  # 五连（胜利，端点在边界也算）
    (5, 2): 100_000,
    (4, 0): 10_000,   # 活四
    (4, 1): 1_000,    # 冲四
    (3, 0): 1_000,    # 活三
    (3, 1): 100,      # 眠三
    (2, 0): 100,      # 活二
    (2, 1): 10,       # 眠二
    (1, 0): 10,       # 活一
    (1, 1): 1,
}

_DIRECTIONS: list[tuple[int, int]] = [(1, 0), (0, 1), (1, 1), (1, -1)]


def get_score(count: int, blocks: int) -> int:
    """根据连子数和封堵端数返回单条线的评分。

    Args:
        count: 连续同色棋子数量。
        blocks: 两端中被对方棋子或边界封堵的端数（0、1 或 2）。

    Returns:
        该棋型的分值；无法形成威胁时返回 0。
    """
    if count >= 5:
        return SCORE_TABLE[(5, 0)]
    if blocks >= 2:
        return 0
    return SCORE_TABLE.get((count, blocks), 0)


def evaluate(board: Board, ai_player: Player) -> int:
    """评估当前棋盘对 ai_player 的净分值。

    采用「起点去重」策略：对每条连续棋型只从起点统计一次，避免重复计分。

    Args:
        board: 当前棋盘状态。
        ai_player: AI 执棋颜色。

    Returns:
        AI 总分 − 对手总分（正值对 AI 有利）。
    """
    opponent = Player.WHITE if ai_player == Player.BLACK else Player.BLACK
    return _score_for(board, ai_player) - _score_for(board, opponent)


def _score_for(board: Board, player: Player) -> int:
    """计算单方棋子的棋盘总分。

    Args:
        board: 当前棋盘状态。
        player: 待评估的一方。

    Returns:
        该方所有棋型分值之和。
    """
    grid = board.grid
    total = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if grid[i][j] != player:
                continue
            for dr, dc in _DIRECTIONS:
                # 只从每条连线的起点开始统计，跳过中间/末端格
                prev_r, prev_c = i - dr, j - dc
                if (
                    0 <= prev_r < BOARD_SIZE
                    and 0 <= prev_c < BOARD_SIZE
                    and grid[prev_r][prev_c] == player
                ):
                    continue

                # 沿方向计数
                count = 0
                r, c = i, j
                while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r][c] == player:
                    count += 1
                    r += dr
                    c += dc

                # 检查末端是否被封堵
                blocks = 0
                if r < 0 or r >= BOARD_SIZE or c < 0 or c >= BOARD_SIZE or grid[r][c] != Player.NONE:
                    blocks += 1
                # 检查起点反方向是否被封堵
                pr, pc = i - dr, j - dc
                if pr < 0 or pr >= BOARD_SIZE or pc < 0 or pc >= BOARD_SIZE or grid[pr][pc] != Player.NONE:
                    blocks += 1

                total += get_score(count, blocks)
    return total
