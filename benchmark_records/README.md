# benchmark_records

本目录同时保留当前正式基线与历史参考记录。公开提交前，优先按下面的层级理解，不要把所有 JSON 一起当成“当前结果”。

## 当前正式基线

当前准备公开提交的正式 `probe2` 基线是：

- `d5_a_white_probe2_25_merged.json`
- `d5_a_black_probe2_25_merged.json`

对应结论：

- `d5_a_white_probe2_25_merged.json`：`16胜 7负 2和`
- `d5_a_black_probe2_25_merged.json`：`25胜 0负 0和`

这些 merged 文件是当前版本最应优先引用的 fixed opening matrix 结果。

## 当前基线的分片与补跑文件

以下文件是生成当前 `probe2` merged 基线时保留的中间结果：

- `d5_a_white_probe2_slice_*.json`
- `d5_a_black_probe2_slice_*.json`
- `d5_a_white_probe2_fill_5_9.json`

保留原因：

- 便于追溯分片运行与补跑来源
- merged 文件中的 `sources` 字段会引用这些文件名

它们不是正式对外结论，但仍有可追溯价值，因此当前保留。

## 旧 baseline

以下文件记录的是 `probe2` 引入前的 fixed opening matrix 基线：

- `d5_a_white_25_merged.json`
- `d5_a_black_25_merged.json`
- `d5_a_white_slice_*.json`
- `d5_a_black_slice_*.json`

它们仍可用于对照 `probe2` 带来的增益，但不应再被当作当前公开基线。

## 历史参考

以下文件来自更早阶段，可用于回顾旧失败模式或生成固定案例：

- `opening_matrix_20260318_*.json`
- `opening_matrix_current.json`
- `opening_matrix_extreme_minimal.json`
- `opening_puzzles_depth5.json`
- `opening_puzzles_depth5.md`
- `white_opening_table_depth5.json`
- `white_opening_table_depth5.md`
- `white_opening_variant_compare.json`
- `d5_a_white_probe_slice_5_10.json`
- `d5_a_black_probe_slice_10_15.json`

这些文件主要用于分析历史问题、核对早期实验，不应直接当作当前版本验收结果。
