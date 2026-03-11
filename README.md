# 五子棋 (Gomoku)

基于 Pygame 的五子棋人机对弈游戏，AI 使用 Minimax + Alpha-Beta 剪枝算法。

## 功能特性

- 15×15 标准棋盘
- 人机对弈，可选执黑（先手）或执白（后手）
- AI 使用 Minimax + Alpha-Beta 剪枝搜索
- 支持悔棋（撤回一轮，即玩家和 AI 各一步）

## 安装

**环境要求：** Python 3.10+

```bash
# 克隆项目
git clone <repo-url>
cd gomoku-test

# 安装运行依赖
pip install pygame

# 或安装完整开发依赖（含测试、格式化、lint 工具）
pip install -e ".[dev]"
```

## 运行

```bash
# 方式一：直接运行单文件原型
python gomoku.py

# 方式二：安装后以模块方式运行
python -m gomoku
```

## 操作说明

| 操作 | 方式 |
|------|------|
| 选择执黑 | 开局界面按 `B` |
| 选择执白 | 开局界面按 `W` |
| 落子 | 鼠标左键点击棋盘交叉点 |
| 悔棋 | 游戏中按 `U`（撤回玩家和 AI 各一步） |
| 重新开始 | 游戏结束后按 `R` |
| 退出 | 游戏结束后按 `Q`，或关闭窗口 |

## 开发

```bash
# 运行测试
pytest tests/ -v

# 代码格式化
black src/ tests/ --line-length 99

# Lint 检查
ruff check src/ tests/
```

## 项目结构

```
gomoku-test/
├── gomoku.py          # 单文件原型（可直接运行）
├── src/gomoku/        # 模块化重构版本（开发中）
│   ├── config.py      # 常量配置
│   ├── board.py       # 棋盘逻辑
│   ├── game.py        # 游戏控制器
│   ├── ai/
│   │   ├── evaluator.py   # 棋面评估
│   │   └── searcher.py    # Minimax 搜索
│   └── ui/
│       └── renderer.py    # Pygame 渲染
└── tests/             # 单元测试
```

## AI 说明

AI 使用经典的 Minimax 算法配合 Alpha-Beta 剪枝：

- **评估函数**：统计双方各方向连子数，按"活四/冲四/活三"等棋型打分
- **搜索深度**：默认 depth=2，可在代码中调整（depth>3 时性能下降明显）
- **候选点生成**：只考虑已有棋子周围 1 格范围内的空位，大幅减少搜索空间
