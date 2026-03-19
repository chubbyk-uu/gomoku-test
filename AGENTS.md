# AGENTS.md

本文件是 `gomoku-test` 当前阶段的唯一协作约束。开始工作前先读它，不要依赖旧聊天记录或过期实验文档。

## 1. 目标

目标是在可复现、正面对战测试里，把 `gomoku-test` 做强，在多种随机开局下对战同级目录下的 `zhou`，无论执黑或者执白都能取得大于60%的胜率：

- 当前仓库：`/home/jerry/python-test/gomoku/gomoku-test`
- 对手引擎：`/home/jerry/python-test/gomoku/zhou`

硬约束：

- 不把 `opening book` / 开局库作为主要修复手段。
- 优先修复根因，不做只掩盖失败现象的补丁。
- `zhou` 基线固定为 `depth=5`。
- `gomoku-test` 正式对抗深度可以是 `4` 或 `5`，上限 `5`。
- 关键开局线上，`depth=5` 不能比 `depth=4` 退化。
- “成功”指棋力变强，不是只变快。

## 2. 当前真实主线

以代码控制流为准，不要只看配置名。

当前主线搜索是：

- `immediate win / immediate block`
- `VCF win / VCF block`
- `iterative deepening minimax`
- `alpha-beta`
- `TT`
- `killer heuristic`
- `local_hotness` 候选排序
- 原生热点加速

当前不在主线里生效的旧东西已经清掉，不要再围绕它们设计实验，包括：
- `quiescence`
- `internal forcing`
- threat-based early return
- root rerank / dynamic cutoff 旧分支

## 3. 当前配置与验证状态

当前 [config.py](/home/jerry/python-test/gomoku/gomoku-test/src/gomoku/config.py) 的关键配置：

- `AI_SEARCH_DEPTH = 5`
- `AI_MAX_CANDIDATES = 20`
- `AI_VCF_ENABLED = True`
- `AI_VCF_MAX_DEPTH = 10`
- `AI_VCF_MAX_CANDIDATES = 16`

当前本地可确认状态：

- `gomoku-test` 当前分支：`mainline-search`
- `PYTHONPATH=src pytest -q tests/test_searcher.py tests/test_benchmark.py` -> `38 passed`
- `gomoku-test` 原生扩展已加载：`_threat_kernels.cpython-311-x86_64-linux-gnu.so`
- `zhou` 原生扩展已加载：`_eval_kernels.cpython-311-x86_64-linux-gnu.so`

说明：

- 对战和 benchmark 默认应在两边原生扩展都可用时进行。
- 纯 Python fallback 的时间数据不用于正式结论。

## 4. 历史参考证据

当前保留的这些记录来自旧主线，生成时的 `minimax` 主链与当前版本不同：

- `benchmark_records/opening_matrix_extreme_minimal.json`
- `benchmark_records/opening_puzzles_depth5.json`
- `benchmark_records/opening_puzzles_depth5.md`
- `benchmark_records/white_opening_table_depth5.json`
- `benchmark_records/white_opening_table_depth5.md`

它们仍然可以用于：

- 回顾过去出现过的失败模式
- 提供固定开局案例
- 帮助定位白棋典型崩点

它们不能直接用于：

- 评估当前版本是否已经提升
- 作为当前版本的正式验收基线

这些历史记录里，最关键的是固定开局矩阵：

- `d5_a_white`: `1 / 24 / 0`
- `d5_a_black`: `2 / 16 / 7`
- `d4_a_white`: `3 / 11 / 11`
- `d4_a_black`: `3 / 14 / 8`

这些历史记录说明旧版本曾经暴露出以下问题：

- 白棋线显著偏弱。
- `depth=5` 比 `depth=4` 更差，存在明确退化。
- 白棋很多失败线的模式是：
  - `W2 = minimax`
  - `W4 = minimax`
  - 很快进入 `vcf_block` / `immediate_block`

因此，当前默认工作假设仍然是：

- 根因更可能在评估、候选排序、深度稳定性。
- `VCF` 更多是在补残局，不是白棋早期失误的首因。

## 5. 当前正式基线

由于 `minimax` 主链已经发生实质变化，已不再对应这些旧记录生成时的版本，旧对局结果不再等同于当前版本基线。

因此：

- 当前正式基线必须由当前代码重新生成。
- 在新的 fixed opening matrix 和随机对战结果生成前，不得把旧记录当作当前版本验收依据。
- 现阶段旧记录只能叫“历史参考证据”，不能叫“当前基线”。

当前优先需要补齐的新基线：

1. 用当前代码重跑 fixed opening matrix
2. 用当前代码重跑小规模随机对战基线
3. 如有必要，再补当前版本的白棋关键开局表

## 6. 基线刷新规则

发生以下任一变化后，必须重刷正式基线：

- `minimax` 主链逻辑变化
- 候选排序逻辑变化
- 评估函数语义变化
- `VCF` 接入顺序或返回条件变化
- 搜索深度语义变化

如果没有重刷，就只能引用“历史参考证据”，不能引用“当前正式基线”。

## 7. 总原则

必须遵守：

- 先复现，再解释，再改代码，再复测。
- 先看固定 opening matrix，再看随机对战。
- 先看白棋失败线，再看黑棋失败线。
- 优先修搜索、评估、候选排序、深度稳定性。
- 任何“变强了”的结论都必须带证据。

默认不接受：

- 引入 opening table 或手工 opening book，针对单个开局点写特判。
- 只调参数、不解释根因。
- 只看总胜率，不看颜色拆分。
- `depth=4` 变好但 `depth=5` 退化。
- 把“更快”说成“更强”。
- 擅自改 `zhou` 语义来制造优势。
- 没有 benchmark 证据时，声称强度提升。
- 依赖某一条“运气好”的自博弈线。

## 8. 优先调查顺序

默认按这个顺序：

1. 确认现象还能复现。
2. 开局阶段 evaluator 偏差
3. 根节点着法排序
4. depth=4 vs depth=5 的  horizon 问题
5. VCF 改进
6. 最后再看速度、节点数、耗时。

如果某个结论无法落到“具体开局、具体手数、具体 trace”，就还不够强。

## 9. 改动策略

优先改：

1. 搜索正确性与深度稳定性
2. 候选排序质量
3. 评估函数对关键开局形状的价值表达
4. `VCF` 与主搜索的边界
5. 性能优化

不优先改：

- opening book
- 大量启发式短路
- 只为降耗时而砍搜索宽度
- 大规模重写但没有可验证假设
- 没有隔离实验支撑的全局分数权重大改
- 大改 VCF,除非确定证明是 VCF 负责
- 那些能帮助黑棋进攻、却伤害白棋结构的宽泛启发式

## 10. 证据要求

### 10.1 声称“变强了”

至少附带：

- fixed opening matrix 前后对比
- 随机对战前后对比
- 按颜色拆分结果
- `depth=4` / `depth=5` 对比

### 10.2 声称“根因在搜索/评估/排序/VCF”

至少附带：

- 具体开局线
- 出错手数
- `trace.source`
- `root_candidates`
- 前后候选或评分差异
- 为什么它能解释 `depth=5` 退化

### 10.3 声称“不是 opening 问题”

至少附带：

- 没有靠硬编码开局表
- 多个同类开局点一起改善
- 改动影响延伸到 move 4/6 之后，而不是只改第一手

## 11. 标准工作流

每次改动至少做完这个闭环：

1. 复现一个失败现象。
2. 提出一个可证伪假设。
3. 一次只做一个聚焦改动，避免大规模重写，保证修改容易解释，容易回归。
4. 跑相关单测。
5. 跑最小必要 benchmark / opening regression。
6. 记录结果是否支持原假设。

如果不支持原假设，不要强行解释，直接回到调查阶段。

## 12. 工具使用

默认工作目录：

- `/home/jerry/python-test/gomoku/gomoku-test`

### 12.1 单元测试

```bash
PYTHONPATH=src pytest -q tests/test_searcher.py
PYTHONPATH=src pytest -q
```

### 12.2 随机对战 benchmark

```bash
PYTHONPATH=src python tools/run_benchmark.py \
  --depth-a 5 \
  --depth-b 5 \
  --repo-b /home/jerry/python-test/gomoku/zhou \
  --games 20 \
  --progress \
  --save-json benchmark_records/some_run.json
```

说明：

- `A` 是当前 `gomoku-test`
- `B` 是 `zhou`
- 颜色随机
- 首手黑棋随机落在中心 `5x5`

### 12.3 固定 opening matrix

```bash
PYTHONPATH=src python tools/run_opening_matrix.py \
  --repo-b /home/jerry/python-test/gomoku/zhou \
  --output-json benchmark_records/opening_matrix_trial.json
```

这是当前最重要的结构化回归工具。

它会固定跑 `100` 盘：

- `depth=5`, A=WHITE, 25 开局
- `depth=5`, A=BLACK, 25 开局
- `depth=4`, A=WHITE, 25 开局
- `depth=4`, A=BLACK, 25 开局

### 12.4 提炼开局案例

```bash
PYTHONPATH=src python tools/extract_opening_puzzles.py \
  some_benchmark.json \
  --output-json benchmark_records/opening_puzzles_depth5.json \
  --output-md benchmark_records/opening_puzzles_depth5.md

PYTHONPATH=src python tools/extract_white_opening_table.py \
  benchmark_records/opening_puzzles_depth5.json \
  --output-json benchmark_records/white_opening_table_depth5.json \
  --output-md benchmark_records/white_opening_table_depth5.md
```

用途：

- 把随机对战中的重复失败模式整理成固定案例。

### 12.5 题库 benchmark

```bash
PYTHONPATH=src python tools/run_puzzle_benchmark.py --depth 4
PYTHONPATH=src python tools/run_puzzle_benchmark.py --depth 5
```

用途：

- 监控战术/判断题退化。
- 不能替代对 `zhou` 的 head-to-head。

### 12.6 VCF benchmark

```bash
PYTHONPATH=src python tools/run_vcf_benchmark.py --depth 8 --repeat 10
```

用途：

- 只验证 `VCF` 子系统本身。
- 不能证明整机变强。

## 13. 文档维护

行为变更后，至少检查是否需要同步更新：

- [README.md](/home/jerry/python-test/gomoku/gomoku-test/README.md)
- [AGENTS.md](/home/jerry/python-test/gomoku/gomoku-test/AGENTS.md)
- 相关测试
- 相关 benchmark 记录

如果本文件与代码现状不一致，直接改本文件，不要把真实状态只留在聊天里。

## 14. 什么才算完成

一项任务只有在以下条件都满足时，才算完成：
- 代码仍然通过相关测试
- 改动已经用通俗语言解释清楚
- benchmark 证据已附上或已总结
- 未解决的不确定性被明确标注
- 如果结果只是部分完成，已经指出下一步最值得做的事情
