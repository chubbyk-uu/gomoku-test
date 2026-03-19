# 五子棋 (Gomoku)

基于 `Pygame` 的五子棋人机对弈项目。AI 当前主线使用模式评估、`Minimax`、`Alpha-Beta` 剪枝、置换表、`VCF`、killer heuristic、`local_hotness` 排序，以及若干 `Cython` 热点加速。

## 当前状态

- 15×15 标准棋盘，黑棋先手
- 支持人机对弈，玩家可选执黑或执白
- 支持悔棋：一次撤回玩家和 AI 各一步
- 支持固定题库回归、搜索 profiling、自对弈 benchmark
- 支持独立 `VCF` benchmark / profiling
- 当前稳定基线：`VCF + minimax + TT + killer + local hotness ordering + white early root probe2 rerank`
- 棋盘带左侧/上方坐标与天元标记，便于定位落点
- 当前默认 AI 配置：
  - 最大搜索深度：`5`
  - 单步时间上限：`None`（仅按最大深度搜索）
  - 候选点上限：`20`
  - 候选邻域半径配置：`2`
  - `VCF` 最大深度：`10`
  - `VCF` 候选上限：`16`

## 主要特性

- `Minimax + Alpha-Beta` 剪枝
- 迭代加深（Iterative Deepening）
- Zobrist 哈希置换表（TT）
- Killer Heuristic
- 评估缓存
- 独立 `VCF` 战术证明器
- 一步成五预检查
- 增量候选点维护
- 增量评估状态缓存
- `Cython` 热点内核（threat / move analysis / line counting）
- 最后一手高亮显示
- 棋盘坐标与天元标记

## 当前基线

当前准备公开提交的正式基线是 `probe2` 版本，也就是：

- 黑白共用主搜索链：`immediate win/block -> VCF win/block -> iterative deepening minimax -> alpha-beta -> TT -> killer -> local_hotness`
- 仅对白棋早期 root 候选启用轻量 `probe2` 二次重排
- 不使用 opening book / 手工特判开局

当前固定 `d5` opening matrix 基线结果：

- 白棋：`16胜 7负 2和`
- 黑棋：`25胜 0负 0和`

对应结果文件：

- [benchmark_records/d5_a_white_probe2_25_merged.json](/home/jerry/python-test/gomoku/gomoku-test/benchmark_records/d5_a_white_probe2_25_merged.json)
- [benchmark_records/d5_a_black_probe2_25_merged.json](/home/jerry/python-test/gomoku/gomoku-test/benchmark_records/d5_a_black_probe2_25_merged.json)

说明：

- 这些 `probe2` merged 文件是当前正式基线。
- `benchmark_records/` 里其余旧 `opening_matrix_*`、`white_opening_*`、`opening_puzzles_*` 等记录属于历史参考，不应再直接当作当前版本验收结果。

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

先编译 `Cython` 扩展，再做任何性能测试或和 `zhou` 的对战对比。

原因：

- 当前搜索大量依赖局部分析、批量 move analysis、`VCF` probe 等热点
- 如果没有原生扩展，运行时会回退到纯 Python 路径
- 纯 Python fallback 的速度和对战结果都不适合拿来做正式 benchmark

简短地说：不先编译 `Cython`，就不要比较性能。

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

AI 搜索器位于 [src/gomoku/ai/searcher.py](/home/jerry/python-test/gomoku/gomoku-test/src/gomoku/ai/searcher.py)。

当前搜索流程大致包括：

- 一步成五预检查
- 一步防输预检查
- `VCF` 必胜 / 防杀预检查
- 迭代加深，从 `depth=1` 逐层加深到最大深度
- 置换表复用历史搜索结果
- killer move 优先级
- 基于 `local_hotness` 的候选排序

当前公开基线额外保留的白棋早期逻辑：

- 仅对白棋早期 root 候选做 `probe2` 二次重排
- `probe2` 只用于白棋早期 root 排序，不改变黑棋主搜索语义

### VCF

`VCF` 求解器位于 [src/gomoku/ai/vcf.py](/home/jerry/python-test/gomoku/gomoku-test/src/gomoku/ai/vcf.py)。

当前实现特性包括：

- 独立 `VCF` 必胜搜索
- 独立 `VCF` 防杀搜索
- 原生批量一步成五检测复用
- `Cython` 原生预筛 `vcf_move_probes`
- 独立 `VCFStats` profiling 统计

当前 `VCF` 采用“两段式”：

- 先用局部原生 probe / threat analysis 拿到强攻种子
- 再对候选真实落子，验证是否直接赢或制造下一步立即赢

### 评估函数

评估器位于 [src/gomoku/ai/evaluator.py](/home/jerry/python-test/gomoku/gomoku-test/src/gomoku/ai/evaluator.py)。

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

### Cython 热点

热点内核位于 [src/gomoku/ai/_threat_kernels.pyx](/home/jerry/python-test/gomoku/gomoku-test/src/gomoku/ai/_threat_kernels.pyx)。

当前已下沉的热点包括：

- 局部 quick pattern summary
- 批量 quick pattern summaries
- 单点 move analysis
- 批量 move analysis
- `VCF` move probes
- 单线 shape counting

性能建议：

- 先执行 `python setup.py build_ext --inplace`
- 再运行题库 benchmark / 自对弈 / 与 `zhou` 的 head-to-head
- 如果没有 `.so` / `.pyd`，先不要解读时间数据

## 项目结构

```text
gomoku-test/
├── src/gomoku/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── board.py
│   ├── game.py
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── _threat_kernels.c
│   │   ├── _threat_kernels.pyx
│   │   ├── evaluator.py
│   │   ├── puzzles.py
│   │   ├── searcher.py
│   │   ├── threats.py
│   │   └── vcf.py
│   └── ui/
│       ├── __init__.py
│       └── renderer.py
├── tests/
│   ├── __init__.py
│   ├── test_benchmark.py
│   ├── test_board.py
│   ├── test_evaluator.py
│   ├── test_game.py
│   ├── test_puzzles.py
│   ├── test_renderer.py
│   ├── test_searcher.py
│   └── test_threats.py
├── tools/
│   ├── benchmark.py
│   ├── engine_worker.py
│   ├── extract_opening_puzzles.py
│   ├── extract_white_opening_table.py
│   ├── run_benchmark.py
│   ├── run_opening_matrix.py
│   ├── run_puzzle_benchmark.py
│   └── run_vcf_benchmark.py
├── benchmark_records/
│   ├── README.md
│   └── ...
└── setup.py
```

## 固定题库

固定题库位于 [src/gomoku/ai/puzzles.py](/home/jerry/python-test/gomoku/gomoku-test/src/gomoku/ai/puzzles.py)，当前覆盖：

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

## VCF Benchmark

运行独立 `VCF` benchmark：

```bash
PYTHONPATH=src python tools/run_vcf_benchmark.py --depth 8 --repeat 10
```

只看某一种模式：

```bash
PYTHONPATH=src python tools/run_vcf_benchmark.py --depth 8 --repeat 10 --mode block
```

输出会包含：

- `avg_time`
- `nodes`
- `cache`
- `attack`
- `defense`
- `prefilter`
- `classify`
- `immediate`
- `depth`

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

## 固定 Opening Matrix

当前正式 head-to-head 基线优先使用 [tools/run_opening_matrix.py](/home/jerry/python-test/gomoku/gomoku-test/tools/run_opening_matrix.py)。

示例：

```bash
PYTHONPATH=src python tools/run_opening_matrix.py \
  --depth-a 5 \
  --depth-b 5 \
  --repo-b /home/jerry/python-test/gomoku/zhou \
  --progress \
  --save-json benchmark_records/some_opening_matrix.json
```

说明：

- `A` 是当前 `gomoku-test`
- `B` 是 `zhou`
- `run_benchmark.py` 更适合随机/自对弈或跨 worktree 对比
- 当前正式结果解读，应优先看 fixed opening matrix，再看随机对战

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
- 游戏状态切换与控制器辅助逻辑
- 渲染器坐标映射
- 评估器棋型识别与增量计数
- 搜索结果稳定性
- TT / 缓存复用
- `VCF` 求解与 benchmark 统计
- 题库回归
- benchmark JSON 输出

## 可调参数

配置文件在 [src/gomoku/config.py](/home/jerry/python-test/gomoku/gomoku-test/src/gomoku/config.py)。

```python
AI_SEARCH_DEPTH = 5
AI_SEARCH_TIME_LIMIT_S = None
AI_MAX_CANDIDATES = 20
AI_CANDIDATE_RANGE = 2
AI_VCF_ENABLED = True
AI_VCF_MAX_DEPTH = 10
AI_VCF_MAX_CANDIDATES = 16
AI_MOVE_DELAY_MS = 10
```

建议：

- 想要更快响应：降低 `AI_SEARCH_DEPTH`，或重新启用 `AI_SEARCH_TIME_LIMIT_S`
- 想要更强棋力：优先改搜索策略/评估/排序，其次再提高深度
- 想减少动画等待：把 `AI_MOVE_DELAY_MS` 调成更小或 `0`
- `AI_CANDIDATE_RANGE` 仍然是有效实现配置，但当前主线不建议把“调宽/调窄候选半径”当作首要优化手段
- `AI_VCF_MAX_DEPTH`、`AI_VCF_MAX_CANDIDATES` 建议结合题库与对弈 benchmark 做 A/B 验证
