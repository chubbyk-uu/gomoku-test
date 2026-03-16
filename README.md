# 五子棋 (Gomoku)

基于 `Pygame` 的五子棋人机对弈项目。AI 使用模式评估、`Minimax`、`Alpha-Beta` 剪枝、置换表与迭代加深搜索，支持固定最大深度和可选限时搜索。

## 当前状态

- 15×15 标准棋盘，黑棋先手
- 支持人机对弈，玩家可选执黑或执白
- AI 默认配置：
  - 最大搜索深度：`4`
  - 单步时间上限：`None`（仅按最大深度搜索）
  - 候选点上限：`15`
- 支持悔棋：一次撤回玩家和 AI 各一步
- 支持 benchmark 自对弈和搜索统计输出

## 主要特性

- `Minimax + Alpha-Beta` 剪枝
- 候选点排序（move ordering）
- 置换表（Transposition Table, Zobrist Hash）
- 评估缓存
- 迭代加深（Iterative Deepening）
- PV 优先搜索
- 一步成五预检查
- 最后一手高亮显示

## 安装

环境要求：`Python 3.10+`

```bash
pip install -e ".[dev]"
```

如果只想运行游戏，安装运行依赖也可以：

```bash
pip install pygame numpy
```

## 运行

推荐从仓库根目录执行：

```bash
PYTHONPATH=src python -m gomoku
```

如果已经 `pip install -e .`，也可以直接运行：

```bash
python -m gomoku
```

## 操作

| 场景 | 按键 / 操作 | 说明 |
|------|-------------|------|
| 开局界面 | `B` | 执黑，玩家先手 |
| 开局界面 | `W` | 执白，AI 先手 |
| 对局中 | 鼠标左键 | 在最近的交叉点落子 |
| 对局中 | `U` | 悔棋（撤回双方各一步） |
| 结束界面 | `R` | 重新开始 |
| 结束界面 | `Q` | 退出游戏 |

## AI 设计

### 搜索

AI 搜索器位于 [src/gomoku/ai/searcher.py](/home/jerry/llm_code_learn/claude_ws/gomoku-test/src/gomoku/ai/searcher.py)。

- 迭代加深从 `depth=1` 逐层加深到最大深度
- 如果设置了时间上限，则在超时前返回最后一层完整搜索结果
- 置换表缓存历史局面的搜索结果
- 上一轮最佳着法会通过 TT/PV 优先在下一轮提前搜索

### 评估函数

评估器位于 [src/gomoku/ai/evaluator.py](/home/jerry/llm_code_learn/claude_ws/gomoku-test/src/gomoku/ai/evaluator.py)。

当前使用模式识别打分，覆盖：

- `FIVE`
- `OPEN_FOUR`
- `HALF_FOUR`
- `OPEN_THREE`
- `HALF_THREE`
- `OPEN_TWO`
- `HALF_TWO`

并包含：

- 组合威胁加分
- 防守权重
- 对局面两方的净分评估

## 项目结构

```text
gomoku-test/
├── src/gomoku/
│   ├── __main__.py
│   ├── config.py
│   ├── board.py
│   ├── game.py
│   ├── ai/
│   │   ├── evaluator.py
│   │   └── searcher.py
│   └── ui/
│       └── renderer.py
├── tests/
│   ├── test_board.py
│   ├── test_evaluator.py
│   └── test_searcher.py
└── tools/
    ├── benchmark.py
    └── run_benchmark.py
```

## Benchmark

运行自对弈 benchmark：

```bash
PYTHONPATH=src python tools/run_benchmark.py --depth-a 4 --depth-b 4 --games 4 --quiet
```

输出内容包括：

- 胜负和平均用时
- 每步平均搜索节点数
- 叶子评估次数
- 排序阶段评估次数
- TT 命中和剪枝统计
- 平均分支因子

## 开发

运行测试：

```bash
pytest -q
```

单测当前覆盖：

- 棋盘落子、悔棋、胜负判断、候选点维护
- 评估器棋型识别
- 搜索结果稳定性
- TT / 缓存复用
- 迭代加深与限时回退
- 超时场景下模拟落子回滚

## 可调参数

配置文件在 [src/gomoku/config.py](/home/jerry/llm_code_learn/claude_ws/gomoku-test/src/gomoku/config.py)。

```python
AI_SEARCH_DEPTH = 4
AI_SEARCH_TIME_LIMIT_S = 3.0
AI_MAX_CANDIDATES = 15
AI_MOVE_DELAY_MS = 100
```

建议：

- 想要更快响应：降低 `AI_SEARCH_TIME_LIMIT_S`
- 想要更强棋力：提高 `AI_SEARCH_DEPTH`，但耗时会明显增加
- 如果只是想减少体感等待：把 `AI_MOVE_DELAY_MS` 调成更小或 `0`
