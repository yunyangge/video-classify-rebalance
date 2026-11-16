# 输入、配置与输出

## 类目与制作形态

类目树 JSON 使用 `primary` 数组。每个节点有 `id/name`；一级可含 `secondaries`，二级可含 `children`。至少提供两个内容中心。示例：

```json
{"primary": [
  {"id": "topic-1", "name": "示例主题一", "secondaries": [
    {"id": "topic-1-detail", "name": "示例子类", "children": []}
  ]},
  {"id": "topic-2", "name": "示例主题二"}
]}
```

制作形态独立提供，至少两项：

```json
{"forms": [
  {"id": "form-1", "name": "示例形态一", "text": "Description of visual form one"},
  {"id": "form-2", "name": "示例形态二", "text": "Description of visual form two"}
]}
```

中心文件在 `centers` 命令中生成。分类前核对配置指纹及向量文件摘要，类目变化后重新生成中心。

## 分类结果

原字段保留，追加以下内容：

| 字段 | 内容 |
|---|---|
| `primary_id/name`、`secondary_id/name` | 命中节点的一级、二级归属 |
| `content_node_id/name/level`、`content_path` | 实际命中的节点和完整路径 |
| `match_score/match_margin` | 内容第一名相似度、与第二名的差值 |
| `production_form_id/name`、`form_score/form_margin` | 制作形态及对应分数 |
| `evidence_empty` | 是否没有有效 tag |

分数表示余弦相似度。文本清洗保留大小写和标点，去掉空项、重复项、占位词及纯链接；拼接文本最长 2048 字符，模型输入最长 512 tokens。中心最长 96 tokens。每卡默认 batch 128，padding 预算为 16,384 tokens。

## 重平衡输入

| 字段 | 要求与用途 |
|---|---|
| `source_uri` | 唯一、非空，关联原记录和最终名单 |
| `primary_id`、`secondary_id`、`production_form_id` | 分类结果；直接命中一级时二级可空 |
| `language` | 原 Meta 的语言字段 |
| `duration_seconds` | 正数，秒 |
| `video_size_bytes` | 非负数，字节 |
| `width/height`、`fps`、`asr` | 像素、帧率、音频采样率 Hz；缺值计入对应技术项的未知数量 |
| `has_audio` | 可选布尔值；提供时参与音频项检查 |
| `pixels`、`upload_timestamp` | 可选排序字段；像素数缺失时用宽×高，时间戳使用秒 |

无有效分类、身份、语言、时长或大小的行输出到 `excluded.parquet`，附 `exclusion_reasons`。重复 `source_uri` 需要在输入前处理。同一份分类表直接作为候选输入，无需添加内部类目编码。

## 目标配置

下列数值只是格式示例：

```json
{
  "target_total_hours": 100,
  "total_tolerance": 0.01,
  "max_gib_per_hour": 2.5,
  "rules": {
    "primary": [{"id": "topic-1", "name": "示例主题一", "min_share": 0.4, "max_share": 0.6}],
    "secondary": [],
    "production_form": [],
    "language": []
  },
  "technical_quality": {
    "min_short_side": 720,
    "min_fps": 24,
    "min_duration_seconds": 10,
    "min_audio_sample_rate_hz": 44100,
    "resolution_shares": [{"min_short_side": 1080, "min_share": 0.5}]
  },
  "lp_seconds": 30,
  "solve_seconds": 120,
  "max_options_per_group": 64,
  "refinement_rounds": 2
}
```

`min_share/max_share` 以目标总小时为分母；`max_share` 可为 null。未列入规则的类目不设置比例目标，但仍计算平均密度。`--seed` 可传一份只含 `source_uri` 的初始名单，求解器可调整它。

## 重平衡输出

| 文件 | 用途 |
|---|---|
| `selected.parquet/json` | 最终选中的原记录，保持原字段 |
| `excluded.parquet` | 缺少必需信息的行及原因 |
| `candidates.parquet` | 本轮可用输入快照，用于重新审核 |
| `quota-summary.json` | 各项目标、实际小时、不足和超额 |
| `technical-quality.json` | 技术达标、未知数量、高分辨率目标 |
| `independent-audit.json` | 唯一性、原值、平均密度及目标审核 |
| `statistics.html`、`hours-summary.json` | 图表与分类小时分布 |
| `report.json`、`solver.json`、`preflight.json`、`config.json` | 耗时、求解状态、输入统计和本轮配置 |

运行输出目录需要是新目录。HTML 自带图表脚本，浏览器本地打开即可。
