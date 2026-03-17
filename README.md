# 五子棋 (Gomoku)

基于 `Pygame` 的五子棋人机对弈项目。AI 目前使用模式评估、`Minimax`、`Alpha-Beta` 剪枝、置换表、威胁分类、短 forcing search，以及若干 `Cython` 热点加速。

## 当前状态

- 15×15 标准棋盘，黑棋先手
- 支持人机对弈，玩家可选执黑或执白
- 支持悔棋：一次撤回玩家和 AI 各一步
- 支持固定题库回归、搜索 profiling、自对弈 benchmark
- 当前默认 AI 配置：
  - 最大搜索深度：`4`
  - 单步时间上限：`None`（仅按最大深度搜索）
  - 候选点上限：`15`
  - 候选邻域半径：`2`

## 主要特性

- `Minimax + Alpha-Beta` 剪枝
- 迭代加深（Iterative Deepening）
- Zobrist 哈希置换表（TT）
- 评估缓存
- 候选点动态截断与两阶段排序
- 威胁分类与短 forcing search
- 一步成五预检查
- 增量候选点维护
- 增量评估状态缓存
- `Cython` 热点内核（threat / move analysis / line counting）
- 最后一手高亮显示

## 安装

环境要求：`Python 3.10+`

推荐先创建虚拟环境，再安装开发依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

如果只想运行游戏，安装运行依赖也可以：

```bash
pip install pygame numpy
```

### 启用 Cython 加速

项目中的 `Cython` 扩展是可选加速层，不影响功能正确性：

- 如果本地扩展已成功编译，运行时会自动加载原生模块
- 如果扩展未编译成功，会自动回退到纯 Python 实现

在新机器上，如需完整编译并启用 `Cython` 加速，通常需要先安装本机编译环境：

- Linux (Debian / Ubuntu)：`python3-dev build-essential`
- Linux (Fedora)：`python3-devel gcc`
- Windows：需要可用的 MSVC Build Tools

安装项目后，可显式构建扩展：

```bash
python setup.py build_ext --inplace
```

如果已经执行过 `pip install -e ".[dev]"`，且本机具备编译环境，通常也会在安装过程中自动构建扩展；上面的命令适合手动重建或显式确认。

### 验证扩展是否已生效

从仓库根目录执行：

```bash
PYTHONPATH=src python -c "import gomoku.ai._threat_kernels as m; print(m.__file__)"
```

如果输出路径指向 `.so`（Linux / macOS）或 `.pyd`（Windows），说明 `Cython` 扩展已生效。

如果导入失败，游戏仍可运行，只是会自动使用纯 Python fallback。

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

当前搜索流程大致包括：

- 迭代加深，从 `depth=1` 逐层加深到最大深度
- 置换表复用历史搜索结果
- 一步成五预检查
- 短 forcing search
- 威胁优先候选生成
- 普通候选的局部粗排、动态截断、精排

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
- 局面净分评估
- 增量线计数缓存

### 威胁分类

威胁分类位于 [src/gomoku/ai/threats.py](/home/jerry/llm_code_learn/claude_ws/gomoku-test/src/gomoku/ai/threats.py)。

当前会识别和使用的核心类别包括：

- `WIN`
- `OPEN_FOUR`
- `DOUBLE_HALF_FOUR`
- `FOUR_THREE`
- `DOUBLE_OPEN_THREE`
- `HALF_FOUR`
- `OPEN_THREE`

### Cython 热点

热点内核位于 [src/gomoku/ai/_threat_kernels.pyx](/home/jerry/llm_code_learn/claude_ws/gomoku-test/src/gomoku/ai/_threat_kernels.pyx)。

当前已下沉的热点包括：

- 局部 quick pattern summary
- 单点 move analysis
- 单线 shape counting

## 项目结构

```text
gomoku-test/
├── src/gomoku/
│   ├── __main__.py
│   ├── config.py
│   ├── board.py
│   ├── game.py
│   ├── ai/
│   │   ├── _threat_kernels.pyx
│   │   ├── evaluator.py
│   │   ├── puzzles.py
│   │   ├── searcher.py
│   │   └── threats.py
│   └── ui/
│       └── renderer.py
├── tests/
│   ├── test_benchmark.py
│   ├── test_board.py
│   ├── test_evaluator.py
│   ├── test_puzzles.py
│   ├── test_searcher.py
│   └── test_threats.py
├── tools/
│   ├── benchmark.py
│   ├── engine_worker.py
│   ├── run_benchmark.py
│   └── run_puzzle_benchmark.py
└── setup.py
```

## 固定题库

固定题库位于 [src/gomoku/ai/puzzles.py](/home/jerry/llm_code_learn/claude_ws/gomoku-test/src/gomoku/ai/puzzles.py)，当前覆盖：

- 一步杀
- 必防冲四
- 主动做活四
- 防双活三
- 防活跳三 + 活三
- 主动做双活三
- 普通中盘连接点
- judgment 类真实/构造局面

运行题库 benchmark：

```bash
PYTHONPATH=src python tools/run_puzzle_benchmark.py --depth 4 --repeat 1
```

只跑某一类题：

```bash
PYTHONPATH=src python tools/run_puzzle_benchmark.py --depth 4 --category judgment
```

## 自对弈 Benchmark

运行本地自对弈 benchmark：

```bash
PYTHONPATH=src python tools/run_benchmark.py --depth-a 4 --depth-b 4 --games 4 --quiet
```

可选参数：

- `--seed`：固定随机种子，便于复现
- `--save-json PATH`：保存每局走子记录和汇总结果
- `--max-moves N`：超过 `N` 手判和，避免长局拖太久

例如：

```bash
PYTHONPATH=src python tools/run_benchmark.py \
  --depth-a 4 --depth-b 4 \
  --games 4 --quiet \
  --seed 7 \
  --max-moves 60 \
  --save-json /tmp/gomoku_selfplay.json
```

### 跨版本对弈

benchmark 还支持不同 `repo/worktree` 对打：

```bash
PYTHONPATH=src python tools/run_benchmark.py \
  --depth-a 3 --depth-b 3 \
  --games 2 --quiet \
  --seed 7 \
  --max-moves 40 \
  --repo-a /path/to/worktree-A \
  --repo-b /path/to/worktree-B \
  --save-json /tmp/gomoku_compare.json
```

这适合做不同 commit 之间的 head-to-head 对比。

## 开发

运行测试：

```bash
pytest -q
```

运行 lint：

```bash
ruff check .
```

单测当前主要覆盖：

- 棋盘落子、悔棋、胜负判断、候选点维护
- 评估器棋型识别与增量计数
- 搜索结果稳定性
- TT / 缓存复用
- 题库回归
- benchmark JSON 输出

## 可调参数

配置文件在 [src/gomoku/config.py](/home/jerry/llm_code_learn/claude_ws/gomoku-test/src/gomoku/config.py)。

```python
AI_SEARCH_DEPTH = 5
AI_SEARCH_TIME_LIMIT_S = None
AI_MAX_CANDIDATES = 15
AI_CANDIDATE_RANGE = 2
AI_MOVE_DELAY_MS = 100
```

建议：

- 想要更快响应：降低 `AI_SEARCH_DEPTH`，或重新启用 `AI_SEARCH_TIME_LIMIT_S`
- 想要更强棋力：优先改搜索策略/威胁处理，其次再提高深度
- 想减少动画等待：把 `AI_MOVE_DELAY_MS` 调成更小或 `0`
- `AI_CANDIDATE_RANGE` 先保持 `2`，除非结合题库和 benchmark 做 A/B 验证
