"""Constants and configuration for Gomoku game."""

from enum import IntEnum


# ============ Player Enum ============
class Player(IntEnum):
    NONE = 0  # 空位
    BLACK = 1  # 黑棋 (先手)
    WHITE = 2  # 白棋


# ============ Game State Enum ============
class GameState(IntEnum):
    MENU = 0  # 颜色选择菜单
    PLAYING = 1  # 对局进行中
    GAME_OVER = 2  # 对局结束


# ============ Board Parameters ============
BOARD_SIZE: int = 15  # 棋盘行列数
GRID_SIZE: int = 40  # 格子间距 (pixels)
MARGIN: int = 20  # 棋盘边距 (pixels)
WINDOW_SIZE: int = MARGIN * 2 + GRID_SIZE * (BOARD_SIZE - 1)  # 窗口边长
FPS: int = 30  # 帧率

# ============ Color Definitions ============
BLACK_COLOR: tuple[int, int, int] = (0, 0, 0)
WHITE_COLOR: tuple[int, int, int] = (255, 255, 255)
BG_COLOR: tuple[int, int, int] = (222, 184, 135)  # 棋盘背景色 (木色)
LINE_COLOR: tuple[int, int, int] = (0, 0, 0)  # 棋盘线条颜色
RED: tuple[int, int, int] = (255, 0, 0)  # 提示文字颜色

# ============ AI Configuration ============
AI_SEARCH_DEPTH: int = 5  # Minimax 最大搜索深度上限
AI_SEARCH_TIME_LIMIT_S: float | None = None  # 可选限时；None 表示仅按最大深度搜索
AI_MAX_CANDIDATES: int = 15  # 每层最多搜索候选点数（move ordering 后截断，减少搜索空间）
AI_CANDIDATE_RANGE: int = 2  # 候选点邻域半径；越大越宽，越小越快但更容易漏点
AI_MOVE_DELAY_MS: int = 100  # AI 落子前的延时 (ms), 便于观察
AI_TT_MAX_SIZE: int = 100_000  # 置换表最大条目数；超限后清空，避免长局无限增长
AI_EVAL_CACHE_MAX_SIZE: int = 100_000  # 评估缓存最大条目数；超限后清空
