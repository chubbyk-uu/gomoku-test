# Gomoku (五子棋) - AI 对弈游戏

## 项目概述
一个基于 Pygame 的五子棋游戏，支持人机对弈。AI 使用 Minimax + Alpha-Beta 剪枝算法。

## 技术栈
- Python 3.10+
- Pygame (UI 渲染)
- 纯 Python 实现 AI 算法 (无 ML 框架依赖)

## 项目结构
```
gomoku/
├── CLAUDE.md              # 本文件 - 项目上下文
├── README.md              # 用户文档
├── requirements.txt       # 依赖
├── pyproject.toml         # 项目元数据
├── src/
│   └── gomoku/
│       ├── __init__.py
│       ├── __main__.py    # 入口: python -m gomoku
│       ├── config.py      # 常量与配置
│       ├── board.py       # 棋盘逻辑 (Board 类)
│       ├── ai/
│       │   ├── __init__.py
│       │   ├── evaluator.py   # 棋面评估
│       │   └── searcher.py    # Minimax 搜索
│       ├── ui/
│       │   ├── __init__.py
│       │   └── renderer.py    # Pygame 渲染
│       └── game.py        # 游戏主循环 (GameController)
└── tests/
    ├── __init__.py
    ├── test_board.py
    ├── test_evaluator.py
    └── test_searcher.py
```

## 编码规范
- 类型提示: 所有函数签名必须有 type hints
- Docstring: Google 风格, 中文注释 + 英文 docstring
- 命名: 类用 PascalCase, 函数/变量用 snake_case, 常量用 UPPER_SNAKE_CASE
- 格式化: black (line-length=99)
- Lint: ruff
- 测试: pytest

## 核心类设计

### Board (board.py)
- 封装棋盘状态 (15x15 二维数组)
- 使用 enum: Player.BLACK=1, Player.WHITE=2, Player.NONE=0
- 方法: place(row, col, player), undo(), check_win(row, col), get_neighbors(), is_full()
- 维护 move_history 支持悔棋

### Evaluator (ai/evaluator.py)
- evaluate(board, ai_player) -> int: 评估棋面分数
- get_score(count, blocks) -> int: 单条线的评分
- 评分表使用常量字典, 便于调优

### Searcher (ai/searcher.py)
- minimax(board, depth, alpha, beta, maximizing, ai_player) -> (score, move)
- 候选点生成与排序优化
- 支持可配置搜索深度

### GameController (game.py)
- 管理游戏状态机: MENU -> PLAYING -> GAME_OVER -> MENU
- 处理用户输入 (落子、悔棋、重启)
- 协调 Board, AI, Renderer

### Renderer (ui/renderer.py)
- draw_board(board): 绘制棋盘和棋子
- draw_menu(): 颜色选择界面
- draw_game_over(winner): 结束画面
- 支持最后一手高亮显示

## AI 优化方向 (后续迭代)
1. 候选点排序 (move ordering): 优先搜索高分位置
2. 迭代加深 (iterative deepening): 在时间限制内逐步加深
3. 置换表 (transposition table): 缓存已评估局面
4. 杀手启发 (killer heuristic): 记住产生截断的好着法
5. 更精细的评分函数: 考虑棋型组合 (如活三+冲四)

## 常用命令
```bash
# 安装依赖
pip install -e ".[dev]"

# 运行游戏
python -m gomoku

# 运行测试
pytest tests/ -v

# 格式化
black src/ tests/ --line-length 99

# Lint
ruff check src/ tests/
```

## 注意事项
- Pygame 初始化和事件循环必须在主线程
- AI 搜索深度 > 3 时需要注意性能, 考虑加时间限制
- 棋盘坐标: board[row][col], row 是行(纵向), col 是列(横向)
- 评分函数的数值平衡很重要, 修改后需要对局测试验证