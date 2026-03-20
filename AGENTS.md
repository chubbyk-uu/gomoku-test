# AGENTS.md

本文件是 `gomoku-test` 当前阶段的唯一协作约束。开始工作前先读它，不要依赖旧聊天记录或过期实验文档。

## 1. 目标

目标是提升真实棋力，在可复现的 head-to-head 测试里尽可能稳定地战胜同级目录下的 `zhou`，以及后续其他五子棋 AI：

- 当前仓库：`/home/jerry/python-test/gomoku/gomoku-test`
- 对手引擎：`/home/jerry/python-test/gomoku/zhou`

硬约束：

- 不把 `opening book` / 手工开局特判作为主要修复手段。
- 优先修根因，不做只掩盖失败现象的补丁。
- `zhou` 基线固定为 `depth=5`。
- `gomoku-test` 正式对抗深度可以是 `4` 或 `5`，上限 `5`。
- 关键开局线上，`depth=5` 不能比 `depth=4` 退化。
- “成功”指棋力变强，不是只变快，也不是只让 benchmark 形态更好看。

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
- early root rerank
- 原生热点加速

当前不在主线里生效的旧东西已经清掉，不要再围绕它们设计实验，包括：

- `quiescence`
- `internal forcing`
- threat-based early return
- 旧版 root rerank / dynamic cutoff 分支

## 3. 当前配置与已确认状态

当前 [config.py](/home/jerry/python-test/gomoku/gomoku-test/src/gomoku/config.py) 的关键配置：

- `AI_SEARCH_DEPTH = 5`
- `AI_MAX_CANDIDATES = 20`
- `AI_CANDIDATE_RANGE = 2`
- `AI_VCF_ENABLED = True`
- `AI_VCF_MAX_DEPTH = 10`
- `AI_VCF_MAX_CANDIDATES = 16`

当前本地可确认状态：

- `gomoku-test` 当前分支：`master`
- `pytest -q` -> `129 passed`
- `TT` 根节点一致性问题已修：根 TT 条目会和最终 root 决策对齐
- root rerank probe 与 `killer` 已解耦：probe 的 reply / stabilizer 排序不再复用主搜索 killer 历史
- 当前 `search` 已直接跟随 `Board.get_candidate_moves()`；修改 `AI_CANDIDATE_RANGE` 会真实影响搜索候选宽度

说明：

- 对战和 benchmark 默认应在两边原生扩展都可用时进行。
- 纯 Python fallback 的时间数据不用于正式结论。
- `benchmark_records/` 现在是本地忽略目录，不再作为仓库基线来源；正式结果需要在工作记录里明确写出日期与命令来源。

## 4. 对 opening matrix 的最新认识

固定 `5x5` center opening matrix 仍然有价值，但必须按“主线簇”解读，不能把 25 盘简单当成 25 个独立样本：

- 当前黑棋 `25/25` 很强，但这 25 盘在当前版本上基本是同一条未触边主线的平移版本。
- 因此，黑棋 `25/25` 不能直接理解成“25 个互相独立的成功案例”，更像一个强主线簇被重复验证。
- 白棋 25 盘也不是 25 个独立问题，而是少数几条归一化主线簇。
- 后续看 fixed opening matrix 时，重点要看：
  - 哪些 opening 属于同一条归一化主线
  - 第一次关键分歧出现在第几手
  - 分歧来自 `minimax`、`vcf_block`、`immediate_block` 还是别的阶段

## 5. 当前最重要的新发现

截至当前工作，已经确认：

- 白棋第 2 手最早的平移分叉，主因不是主 `minimax` 搜索，而是 root rerank probe 错误复用 `killer` 历史；这一点已经修复。
- 修完后，白棋 fixed opening matrix 仍然会出现明显簇状分界，但分歧点被收缩到了更具体的位置。
- 当前代表性白棋赢簇和输簇，在前 5 手归一化后完全一致。
- 第一次关键分歧出现在第 6 手，且 `trace.source` 是 `vcf_block`。
- 因此，当前最该查的问题不再是“root rerank 是否复用 killer”，而是：
  - 为什么白棋在靠边 opening 簇上，会在第 6 手 `vcf_block` 分成两条不同主线
  - 这个分歧是由模拟分支边界效应、VCF 候选生成 / 排序 / 截断，还是其他固定棋盘偏置导致

额外说明：

- `center_bias` 仍存在于局部 move analysis / VCF 局部分析中。
- 但目前还没有足够证据证明它是当前白棋主问题的首要根因。
- 在没有更强证据前，不要先做“加大 center bias”这类全局改动。

## 6. 当前重点调查顺序

默认按这个顺序：

1. 复现当前白棋赢簇 / 输簇的归一化代表局面
2. 对比它们在第 6 手 `vcf_block` 的内部候选、返回理由和 trace
3. 判断分歧是否来自模拟分支边界效应
4. 判断分歧是否来自 VCF 候选生成 / 排序 / 截断
5. 只在拿到这一步证据后，再决定要不要处理 `center_bias` 或其他固定棋盘偏置
6. 最后再看更大规模 benchmark、速度、节点数、耗时

如果某个结论无法落到“具体 opening 簇、具体手数、具体 trace.source、具体候选列表”，就还不够强。

## 7. 总原则

必须遵守：

- 先复现，再解释，再改代码，再复测。
- 先看固定 opening matrix 的归一化主线，再看随机对战。
- 先看白棋失败簇，再看黑棋主线。
- 优先修搜索正确性、VCF 边界、候选排序和深度稳定性。
- 任何“变强了”的结论都必须带证据。

默认不接受：

- 引入 opening table 或手工 opening book，针对单个开局点写特判。
- 只调参数、不解释根因。
- 只看总胜率，不看颜色拆分。
- 把“未触边的 25 盘平移重复样本”当成 25 个独立成功证据。
- `depth=4` 变好但 `depth=5` 退化。
- 把“更快”说成“更强”。
- 擅自改 `zhou` 语义来制造优势。
- 没有 benchmark 证据时，声称强度提升。

## 8. 改动策略

优先改：

1. 搜索正确性与深度稳定性
2. `VCF block` / `VCF win` 在关键 opening 簇上的正确性
3. 候选排序质量
4. 评估函数对关键开局形状的价值表达
5. 性能优化

不优先改：

- opening book
- 大量启发式短路
- 只为降耗时而砍搜索宽度
- 没有隔离实验支撑的全局分数权重大改
- 在没有证据前先增强 `center_bias`

## 9. 证据要求

### 9.1 声称“变强了”

至少附带：

- fixed opening matrix 前后对比
- 随机对战前后对比
- 按颜色拆分结果
- `depth=4` / `depth=5` 对比

### 9.2 声称“根因在搜索 / VCF / 评估 / 排序”

至少附带：

- 具体 opening 簇
- 具体出错手数
- `trace.source`
- `root_candidates` 或 `VCF` 候选
- 前后候选或评分差异
- 为什么它能解释这条归一化主线为何分叉

## 10. 标准工作流

每次改动至少做完这个闭环：

1. 复现一个失败簇或分歧手。
2. 提出一个可证伪假设。
3. 一次只做一个聚焦改动，避免大规模重写，保证修改容易解释，容易回归。
4. 跑相关单测。
5. 复测对应 opening 簇或最小 benchmark。
