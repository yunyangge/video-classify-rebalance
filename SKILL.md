---
name: video-classify-rebalance
description: 操作本仓库，对单份视频 Meta 进行 Qwen3-Embedding-4B 分类、按时长配额重平衡，并核对名单和统计报告。适用于运行、排查或修改这套代码。
---

# 视频 Meta 分类与重平衡

在本仓库根目录执行命令。根据当前任务选择分类、重平衡或审核，不需要每次重跑全部步骤。

## 先确定输入

- 分类：Meta Parquet、类目树 JSON、制作形态 JSON、本地 Qwen3-Embedding-4B 权重。已有中心目录可直接复用。
- 重平衡：带分类、语言、时长、大小的 Parquet，以及本轮目标配置 JSON。可选 seed 是只含 `source_uri` 的初始名单。
- 审核：已有重平衡输出目录。

字段、单位和配置格式查 [docs/input-output.md](docs/input-output.md)；求解规则查 [docs/rebalancing.md](docs/rebalancing.md)。使用用户给定的数据和配置；示例生成器用于演示，不作为业务配额来源。Meta、权重和运行输出保存在仓库外。

## 准备环境

```bash
python -m pip install -e .
python -m video_pipeline --help
```

分类需要可用的 PyTorch、Transformers 和本地模型。优先复用已验证环境；安装示例与版本见 [README.md](README.md)，本地和 RTX 5090 的验证记录见 [docs/testing.md](docs/testing.md)。

## 分类

首次使用某套类目时建立中心；类目发生变化后重新生成到新目录：

```bash
python -m video_pipeline centers \
  --config /data/taxonomy.json \
  --model /data/Qwen3-Embedding-4B \
  --output /data/centers

python -m video_pipeline classify \
  --input /data/meta.parquet \
  --config /data/taxonomy.json --centers /data/centers \
  --forms /data/forms.json --model /data/Qwen3-Embedding-4B \
  --device cuda:0 --output /data/classified
```

`/data/...` 替换为实际路径。中心目录与类目配置必须匹配。需要先检查输入和配置时，在分类命令后加 `--validate-only`；此时不加载模型。

处理方式：每条视频的整组 tag 清洗、去重后按原顺序拼接，编码一次，得到完整 2560 维向量，再分别匹配内容和制作形态。查询上限 512 tokens，中心上限 96 tokens。直接命中一级或二级节点时保留该结果；空 tag 标为 `unclassified`。分数是相似度。

输出检查：`classified.parquet/json` 保留原字段并追加分类结果；`report.json` 记录完成状态、条数和参数；`statistics.html` 与 `hours-summary.json` 展示分类小时分布。只有需要改变显存或速度时才调整 `--batch-size`、`--max-padded-tokens`，并记录本轮取值。

## 重平衡

```bash
python -m video_pipeline rebalance \
  --input /data/classified/classified.parquet \
  --config /data/rebalance.json --output /data/selected
```

可加 `--validate-only` 检查候选和配置，或加 `--seed /data/seed.parquet` 提供初始名单。使用新的输出目录。

处理方式：总时长、类目/形态/语种比例、技术指标、高分辨率比例依次优化，再按密度与质量排序。比例按固定目标小时计算。每条视频最多入选一次；总体及各内容类目、制作形态的平均 `总 GiB ÷ 总小时` 不超过配置上限。单条高密度视频可以和低密度视频搭配。

输出检查：

- `selected.parquet/json`：最终名单。
- `excluded.parquet`：缺少必要信息的记录及原因。
- `quota-summary.json`：实际小时、目标区间、不足与超额。
- `technical-quality.json`：技术指标、未知数量和分辨率目标。
- `statistics.html`：交付量、目标差距和分类分布，页面可下载对应 JSON。

## 重新审核

```bash
python -m video_pipeline audit --run /data/selected
```

查看 `independent-audit.json`：`hard_constraints_passed` 表示唯一性、原始字段和平均密度通过；`quota_requirements_passed`、`technical_quality_passed` 分别表示配额与技术目标满足；`passed` 表示全部满足。若只是 `completed=true`，继续读具体缺口。时间预算用尽后保留可行名单，并在 `solver.json` 中记录各阶段状态。

## 演示与修改代码

不用模型的演示：

```bash
python examples/generate_demo.py /tmp/video-demo
python -m video_pipeline rebalance \
  --input /tmp/video-demo/classified.parquet \
  --config /tmp/video-demo/rebalance.json --output /tmp/video-selected
```

修改代码后运行：

```bash
python -m unittest discover -s tests -v
```

保留整组 tag 编码、完整维度、节点直接命中、固定目标小时和平均密度的计算方式。报告性能时写清数据规模、设备数量、模型是否真实运行，以及是否包含加载和导出时间。
