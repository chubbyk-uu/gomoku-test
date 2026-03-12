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
AI_SEARCH_DEPTH: int = 3  # Minimax 搜索深度 (候选点排序后 3 层可接受)
AI_MOVE_DELAY_MS: int = 500  # AI 落子前的延时 (ms), 便于观察
