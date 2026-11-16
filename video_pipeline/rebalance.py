"""Run balanced selection over one classified metadata table."""

import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .audit import audit
from .common import new_directory, write_json, export_json
from .data import Data, prepare, read_config
from .selection import select
from .report import render


def run(args):
    config = read_config(args.config)
    table, excluded = prepare(args.input)
    data = Data(table, config)
    seed = np.zeros(data.n, dtype=bool)
    if args.seed:
        requested = pq.read_table(args.seed, columns=["source_uri"])["source_uri"].to_pylist()
        lookup = {uri: index for index, uri in enumerate(table["source_uri"].to_pylist())}
        if len(set(requested)) != len(requested) or not set(requested) <= lookup.keys():
            raise ValueError("Seed must contain unique source_uri values from the candidate pool")
        seed[[lookup[uri] for uri in requested]] = True
    preflight = dict(input_rows=len(table) + len(excluded), candidate_rows=len(table),
                     excluded_rows=len(excluded), seed_rows=int(seed.sum()))
    if args.validate_only:
        return preflight
    out = new_directory(args.output)
    started = time.perf_counter()
    pq.write_table(table, out / "candidates.parquet", compression="zstd")
    pq.write_table(excluded, out / "excluded.parquet", compression="zstd")
    write_json(out / "config.json", config)
    write_json(out / "preflight.json", preflight)
    indices, solver = select(data, config, seed)
    final = table.take(pa.array(indices, type=pa.int64()))
    pq.write_table(final, out / "selected.parquet", compression="zstd")
    export_json(out / "selected.parquet", out / "selected.json")
    write_json(out / "solver.json", solver)
    report = dict(completed=True, **preflight, selected_rows=len(final),
                  target_hours=config["target_total_hours"], solver=solver,
                  selection_seconds=time.perf_counter() - started)
    write_json(out / "report.json", report)
    result = audit(out)
    report.update(result)
    report["seconds"] = time.perf_counter() - started
    write_json(out / "report.json", report)
    import json
    stats = json.loads((out / "hours-summary.json").read_text())
    quotas = json.loads((out / "quota-summary.json").read_text())["checks"]
    quality = json.loads((out / "technical-quality.json").read_text())["checks"]
    labels = {rule["id"]: rule.get("name", rule["id"])
              for rules in config["rules"].values() for rule in rules}
    render(out / "statistics.html", "均衡重采样", report, stats, labels=labels,
           quotas=quotas, quality=quality, densities=result["density_checks"], cap=config["max_gib_per_hour"])
    return report
