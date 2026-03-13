# Gomoku (五子棋) - AI 对弈游戏

## 项目概述
一个基于 Pygame 的五子棋游戏，支持人机对弈。AI 使用 Minimax + Alpha-Beta 剪枝算法，配合候选点排序和组合棋型评估。

## 技术栈
- Python 3.10+
- Pygame (UI 渲染)
- 纯 Python 实现 AI 算法 (无 ML 框架依赖)

## 项目结构
```
gomoku/
├── CLAUDE.md              # 本文件 - 项目上下文
├── README.md              # 用户文档
├── pyproject.toml         # 项目元数据与工具配置
├── src/
│   └── gomoku/
│       ├── __init__.py
│       ├── __main__.py    # 入口: python -m gomoku
│       ├── config.py      # 常量与配置 (Player/GameState 枚举, 尺寸, 颜色, AI 参数)
│       ├── board.py       # 棋盘逻辑 (Board 类)
│       ├── ai/
│       │   ├── __init__.py
│       │   ├── evaluator.py   # 棋面评估 (含组合棋型 + 防守加权)
│       │   └── searcher.py    # Minimax 搜索 (含候选点排序)
│       ├── ui/
│       │   ├── __init__.py
│       │   └── renderer.py    # Pygame 渲染
│       └── game.py        # 游戏主循环 (GameController 状态机)
└── tests/
    ├── __init__.py
    ├── test_board.py      # 14 个测试用例
    ├── test_evaluator.py  # 13 个测试用例
    └── test_searcher.py   # 5 个测试用例
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
- 方法: place(row, col, player), undo(), check_win(row, col), get_candidate_moves(), is_full(), copy()
- 维护 move_history 和 last_move 支持悔棋和高亮

### Evaluator (ai/evaluator.py)
- evaluate(board, ai_player) -> int: AI总分 - 对手总分×1.5（防守加权）
- get_score(count, blocks) -> int: 单条线评分，查 SCORE_TABLE
- _score_for(board, player) -> int: 单方总分，含组合棋型加分
- 组合加分: 双活三 +5000，活三+冲四 +5000
- DEFENSE_WEIGHT = 1.5: 对手威胁额外放大

### Searcher (ai/searcher.py)
- AISearcher(depth=3, ai_player): 默认深度 3
- find_best_move(board) -> Optional[tuple]: 不修改传入棋盘
- _minimax: 搜索前对候选点按即时评分排序（Move Ordering），提升剪枝效率

### GameController (game.py)
- 状态机: MENU → PLAYING → GAME_OVER → MENU
- run() 为主循环入口，每帧调 _tick() 分发事件
- 悔棋撤销最多 2 步，回到玩家回合

### Renderer (ui/renderer.py)
- draw_board(board): 背景+网格+棋子+最后一手高亮
- draw_menu(): 开局颜色选择界面
- draw_game_over(winner_text): 半透明叠加结果框
- pixel_to_board(pos): 鼠标坐标→棋盘坐标
- 不调用 display.flip()，由 GameController 统一刷新

## AI 已实现优化
1. ✅ 候选点排序 (Move Ordering): 搜索前按即时评分排序
2. ✅ Alpha-Beta 剪枝: 大幅减少搜索节点
3. ✅ 搜索深度 3: 有排序加持后性能可接受
5. ✅ 防守加权: 对手威胁×1.5

## AI 后续优化方向
- 迭代加深 (iterative deepening): 在时间限制内逐步加深
- 置换表 (transposition table): 缓存已评估局面
- 杀手启发 (killer heuristic): 记住产生截断的好着法

## 常用命令
```bash
# 安装依赖
pip install -e ".[dev]"

# 运行游戏
PYTHONPATH=src python -m gomoku

# 运行测试
pytest tests/ -v

# 格式化
black src/ tests/ --line-length 99

# Lint
ruff check src/ tests/
```

## 注意事项
- Pygame 初始化和事件循环必须在主线程
- AI 搜索深度 > 4 时需要注意性能，考虑加时间限制
- 棋盘坐标: board[row][col], row 是行(纵向), col 是列(横向)
- pytest 运行需要 pyproject.toml 中的 pythonpath=["src"] 和 --import-mode=importlib
- 评分函数的数值平衡很重要，修改后需要对局测试验证

## 当前进度（截至 2026-03-13）

### 已完成的 Bug 修复
1. **`game.py` run() 主循环重复初始化**（已修复）
   - 原代码在 MENU 状态下每帧都调用 `_start_new_game()`，导致棋盘状态被反复重置
   - 修复方案：引入 `prev_state`，仅在 `GAME_OVER → MENU` 状态转换时才触发 `_start_new_game()`

2. **`searcher.py` 置换表 minimizing 分支 flag 错误**（已修复）
   - 原代码在 minimizing 无截断路径无条件存 `"E"`（精确值），但当所有子节点分值均 ≥ 调用方传入的 `beta` 时，真实值是下界，应存 `"L"`
   - 修复方案：保存 `beta_orig`，用 `"L" if best_score >= beta_orig else "E"` 正确区分

### 已实现的 AI 优化（更新）
1. ✅ 候选点排序 (Move Ordering)
2. ✅ Alpha-Beta 剪枝
3. ✅ 置换表 (Transposition Table)：Zobrist 哈希 + fail-soft flag（`"E"`/`"L"`/`"U"`）
4. ✅ 增量候选点集合：`_candidates` + `_candidate_history`，undo 精确回滚
6. ✅ 防守加权：对手威胁×1.5

### 性能教训：numpy 不适合此场景
- 尝试过将 `_score_for` 改写为 numpy 向量化版本（`np.diag` + `np.diff` + `np.where`）
- 结果：速度变慢约 18 倍。原因：numpy 对 5–15 元素的小数组每次调用有 2–5 μs 固定开销，全局约 720 μs vs 原版 ~40 μs
- 结论：**已回退到纯 Python 实现**；若要进一步加速评估，正确方向是**增量评估**（仅重算最后一手影响的若干条线），而非全局 numpy 向量化

### 已知待改进项
- **迭代加深**：可在时间预算内逐步加深，尚未实现
- **杀手启发**：记住产生截断的着法，尚未实现
- **增量评估**：`_score_for` 每次全盘扫描，可改为只重算受最后一手影响的行/列/对角线
- **棋形评估**：尚未对双活四、双冲四、活四＋冲四进行精确判断
