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
- `gomoku-test` 正式对抗深度可以是可变，默认 `5`。
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
- `pytest -q` 应保持全绿；当前主线在调整为“黑开白关 rerank”后，相关 rerank 测试需要显式打开 white rerank 再验证
- `TT` 根节点一致性问题已修：根 TT 条目会和最终 root 决策对齐
- root rerank probe 与 `killer` 已解耦：probe 的 reply / stabilizer 排序不再复用主搜索 killer 历史
- `native / fallback` 关键语义已对齐：不应再因为是否编译原生扩展而改变 `VCF` 攻击候选集合或 `prefilter` 顺序
- `VCF` 已有结构化 trace：`VCFSolver.last_trace`
- 当前 `search` 已直接跟随 `Board.get_candidate_moves()`；修改 `AI_CANDIDATE_RANGE` 会真实影响搜索候选宽度

说明：

- 对战和 benchmark 默认应在两边原生扩展都可用时进行。
- 纯 Python fallback 的时间数据不用于正式结论。
- `benchmark_records/` 现在是本地忽略目录，不再作为仓库基线来源；正式结果需要在工作记录里明确写出日期与命令来源。

## 4. 对 opening matrix 的最新认识

旧的 `5x5` center opening matrix 仍然有分析价值，但不能再当默认正式基线。当前更推荐的快速固定开局集是 5 点：

- 天元 `(7,7)`
- 左上 `(4,4)`
- 右上 `(4,10)`
- 左下 `(10,4)`
- 右下 `(10,10)`

原因：

- 旧 `25` 点矩阵对黑棋存在大量“同一条未触边主线的平移重复”，样本独立性不足。
- 新 `5` 点集更适合快速观察中心与四角方向差异，能更快暴露边界相关问题。

对旧 `25` 点 center matrix 的结论仍然成立，但必须按“主线簇”解读，不能把 25 盘简单当成 25 个独立样本：

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
- 当前代表性赢簇 / 输簇在第 6 手前，顶层 `VCF` 都会看到一对归一化等价强攻；问题不是“只有一个正确防点”。
- 现在已经进一步确认：`_generate_blocking_moves()` 会把两个几乎等价的可行防点排成不同顺序，而 `find_blocking_move()` 使用“首个成功即返回”，会把这个小排序差异直接放大成不同主线。
- 该问题修复后，`d5` 对 `zhou` 的白棋 fixed opening matrix 已从 `10胜 15负 0和` 提升到 `20胜 5负 0和`；剩余失败全部收缩到最左一列 opening。
- 左列剩余失败现在不再是第 6 手 `vcf_block` 分叉；代表性输局与相邻赢局前 7 手归一化后完全一致，第一个分歧已推迟到第 8 手 `minimax`。
- 对左列代表失败局第 8 手做强制候选实验后已确认：
  - 原本看起来更自然的旧路 `normalized (7,3)` 在左边界下确实更差，会更快输给 `zhou`
  - 但当前实战手也不是最优，因为存在替代手 `normalized (4,4)`，强制该手后对白 `zhou(depth=5)` 可走成胜线
- 对该第 8 手局面的实际棋盘候选继续拆解后，当前以 `AISearcher(5, Player.WHITE).find_best_move()` 的 depth-5 最终 root trace 为准：
  - 实际可赢 `zhou` 的候选是实际坐标 `(4,2)`
  - 在同一实际局面的 depth-5 最终 root trace 里，`(4,2)` 的原始根分是 `568`，属于原始前排候选
  - 但 early root rerank 会把 `(4,2)` 压到最终第 `5`，并把根决策改成 `(4,4)`
  - 因此“为什么没选到胜手”当前至少有一层已坐实：root rerank 确实把实际可赢候选 `(4,2)` 从前排压到了后面
  - 后续再分析原始 `minimax` 时，必须使用同一实际局面、同一 depth=5、同一 `find_best_move()` 口径；不要混用归一化坐标或直接 `_minimax()` 的其他口径
- 对 `(4,4)` 这条被 rerank 抬到第一的线，已坐实一个更具体的 white-rerank bug：
  - 黑方 reply `(5,7)` 之后，白方 stabilizer `(3,5)` / `(7,1)` 会在静态评估里形成 `OPEN_FOUR`，旧逻辑因此给出约 `48428` 的高分
  - 但该局面轮到黑走时，黑方其实有 `immediate_win`
  - 这类“白方静态大优但黑下一手直接赢”的假好 stabilizer，会把 `(4,4)` 的 `avg_reply_score` 人为拉到极端负值
  - 当前已修正：rerank probe 里，若 stabilizer 后对手下一手有 `immediate_win`，该 stabilizer 直接记成极差，不再按高静态分记好
  - 该修正把 `(4,4)` 在关键局面上的 `avg_reply_score` 从 `-16444` 收缩到 `-644`，同时没有破坏黑棋 5 开局 `5胜 0负`
- 但这一步不是最终解：
  - 修完后 `(4,4)` 在该关键局面里仍然排第 `1`
  - 白棋真正剩余的主问题仍是 reply source：黑方真实关键 reply `(7,4)` 在候选池里、在黑方完整 depth-5 搜索前排里，但在白方 rerank probe 的 top-3 reply 里仍然缺失
- 当前 5 开局快速基线（黑白都开 rerank）：
  - 白棋：`2胜 3负`
  - 黑棋：`5胜 0负`
- 关闭 rerank 做对照后，5 开局结果会变成：
  - 白棋：`4胜 1负`
  - 黑棋：`2胜 3负`
- 当前最好的实用组合已经确认是：
  - 黑棋 rerank：开
  - 白棋 rerank：关
  - 对应 5 开局结果：白棋 `4胜 1负`、黑棋 `5胜 0负`
- 所以当前不能简单“全关 rerank”，也不该继续默认“黑白都开 rerank”：
  - rerank 确实会误伤某些白棋边界局面
  - 但它同时对黑棋整体强度贡献很大
  - 现在更像是 rerank 的收益/伤害高度依赖颜色和开局簇
- 最近又补了一条更保守、可保留的 probe 修正：
  - rerank probe 现在至少识别“当前轮到谁、当前手是否已有 immediate win”
  - 这条 side-aware immediate-race 修正没有改善白棋 5 开局，但也没有伤到黑棋 5 开局
  - 因此它可以保留为 correctness 修正，但不要把它误当成白棋 rerank 的主解
- `25` 点 center matrix 现在更适合做归一化主线/分歧手研究。
- `5` 点固定 opening 集更适合做日常快速回归。
- 因此，当前最该查的问题不再是“root rerank 是否复用 killer”，而是：
  - 为什么第 8 手 `find_best_move(depth=5)` 的最终根重排会把已经被实验验证可赢 `zhou` 的实际候选 `(4,2)` 压到第 `5`
  - 为什么 rerank 会改选 `(4,4)` 而不是保留 `(4,2)` 这种真实可赢候选
  - 为什么白方 rerank 的 reply source 仍然会漏掉像 `(7,4)` 这样真实关键、但不够 `local_hotness` 的黑方 reply
  - 在统一到同一实际局面和同一 depth-5 口径后，再继续判断原始 `minimax` 与 rerank 各自承担多少责任

额外说明：

- `center_bias` 仍存在于局部 move analysis / VCF 局部分析中。
- 但目前还没有足够证据证明它是当前白棋主问题的首要根因。
- 在没有更强证据前，不要先做“加大 center bias”这类全局改动。

## 6. 当前重点调查顺序

默认按这个顺序：

1. 复现当前左列剩余失败簇的归一化代表局面
2. 对第 8 手局面，先固定到同一实际棋盘和同一 depth=5 `find_best_move()` 口径，避免混用归一化坐标或直接 `_minimax()` 口径
3. 再拆 early root rerank 的 reply/stabilizer probe，解释为什么 `(4,2)` 会被从前排压到第 `5`
4. 在统一口径后，再继续验证 white rerank 的对手 reply top-k 是否漏掉真实关键 reply，以及即使纳入关键 reply 后，probe 评分语义为何仍会误判
5. 判断 static eval 在边界 forcing 结构上的表达盲点是否在放大这类误判
6. 把已修复的第 6 手 `vcf_block` 多解问题只当背景，不再当当前第一调查点
7. 再判断分歧里是否还有模拟分支边界效应或其他固定棋盘偏置
8. 只在拿到这一步证据后，再决定要不要处理 `center_bias`
9. 最后再看更大规模 benchmark、速度、节点数、耗时

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

当前热点分析已确认：

- 现有 `VCF` benchmark 下，最重的不是 `prefilter`，而是 `immediate_win_checks`
- 第二热点是 `classify_attack_moves()` / exact classification
- 所以后续提速优先看 `_find_immediate_wins()` 与 `classify_attack_moves()` / `_count_shapes_after_move()`，不要再让 `prefilter` 承担近似剪枝语义

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
