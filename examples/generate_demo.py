"""Generate synthetic metadata and configuration; no model inference."""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def generate(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    for i in range(24):
        rows.append(dict(sample_id=f"demo-{i:03d}", source="synthetic",
                         source_uri=f"https://example.invalid/video/{i}", tag=["demo", f"topic-{i % 2}"],
                         primary_id=f"topic-{i % 2}", secondary_id=None,
                         production_form_id=f"form-{(i // 2) % 2}", language="demo-language",
                         duration_seconds=900., video_size_bytes=int((.35 + .05 * (i % 8)) * 2**30),
                         width=2560 if i % 3 == 0 else 1920,
                         height=1440 if i % 3 == 0 else 1080,
                         fps=30., asr=48000., upload_timestamp=1000 + i))
    pq.write_table(pa.Table.from_pylist(rows), folder / "classified.parquet")
    taxonomy = {"primary": [{"id": f"topic-{i}", "name": f"示例主题 {i + 1}"} for i in range(2)]}
    forms = {"forms": [{"id": f"form-{i}", "name": f"示例形态 {i + 1}", "text": f"Example visual form {i + 1}"} for i in range(2)]}
    config = dict(target_total_hours=3., total_tolerance=0, max_gib_per_hour=2.5,
                  solve_seconds=10, lp_seconds=3, refinement_rounds=1,
                  rules={"primary": [{"id": f"topic-{i}", "name": f"示例主题 {i + 1}", "min_share": .4, "max_share": .6} for i in range(2)],
                         "production_form": [{"id": f"form-{i}", "name": f"示例形态 {i + 1}", "min_share": .4, "max_share": .6} for i in range(2)]},
                  technical_quality=dict(min_short_side=720, min_fps=24, min_duration_seconds=10,
                                         min_audio_sample_rate_hz=44100,
                                         resolution_shares=[dict(min_short_side=1080, min_share=.5), dict(min_short_side=1440, min_share=.05)]))
    for name, value in [("taxonomy.json", taxonomy), ("forms.json", forms), ("rebalance.json", config)]:
        (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(folder)


if __name__ == "__main__":
    generate(sys.argv[1])
