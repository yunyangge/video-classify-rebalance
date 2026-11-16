"""Validate external selection rules and prepare a single candidate collection."""

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .common import DIMENSIONS, put
from .quality import measurements, numbers


def read_config(path):
    config = json.loads(Path(path).read_text())
    for key in ["target_total_hours", "max_gib_per_hour"]:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    tolerance = config.get("total_tolerance", 0)
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("total_tolerance must be between 0 and 1")
    config["total_tolerance"] = tolerance
    rules = config.setdefault("rules", {})
    for dimension in DIMENSIONS:
        items = rules.setdefault(dimension, [])
        ids = [rule["id"] for rule in items]
        if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{dimension} rule IDs must be unique strings")
        for rule in items:
            low, high = rule.get("min_share", 0), rule.get("max_share")
            if not isinstance(low, (int, float)) or not math.isfinite(low) or not 0 <= low <= 1:
                raise ValueError(f"Invalid minimum share for {rule['id']}")
            if high is not None and (not isinstance(high, (int, float)) or not math.isfinite(high) or not low <= high <= 1):
                raise ValueError(f"Invalid maximum share for {rule['id']}")
    quality = config["technical_quality"]
    for key in ["min_short_side", "min_fps", "min_duration_seconds", "min_audio_sample_rate_hz"]:
        value = quality[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"technical_quality.{key} must be positive and finite")
    sizes = []
    for rule in quality["resolution_shares"]:
        size, share = rule["min_short_side"], rule["min_share"]
        if not isinstance(size, (int, float)) or not math.isfinite(size) or size <= 0:
            raise ValueError("Resolution thresholds must be positive and finite")
        if not isinstance(share, (int, float)) or not math.isfinite(share) or not 0 <= share <= 1:
            raise ValueError("Resolution shares must be between 0 and 1")
        sizes.append(size)
    if len(sizes) != len(set(sizes)):
        raise ValueError("Resolution thresholds must be unique")
    for key, default in [("lp_seconds", 30), ("solve_seconds", 120)]:
        value = config.setdefault(key, default)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key, default, low, high in [("max_options_per_group", 64, 8, 4096), ("refinement_rounds", 2, 1, 4)]:
        value = config.setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key} must be an integer in [{low}, {high}]")
    return config


def prepare(path):
    table = pq.read_table(path)
    required = {"source_uri", "primary_id", "secondary_id", "production_form_id", "language",
                "duration_seconds", "video_size_bytes"}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError("Input missing columns: " + ", ".join(sorted(missing)))
    for name in ["source_uri", *DIMENSIONS.values()]:
        if not (pa.types.is_string(table[name].type) or pa.types.is_large_string(table[name].type) or pa.types.is_null(table[name].type)):
            raise ValueError(f"{name} must contain strings")
    uri = table["source_uri"].to_pylist()
    nonempty = [value for value in uri if value is not None and value.strip()]
    if len(nonempty) != len(set(nonempty)):
        raise ValueError("source_uri must be unique; remove repeated records before selection")
    reasons = [[] for _ in range(len(table))]
    for name in ["source_uri", "primary_id", "production_form_id", "language"]:
        for index, value in enumerate(table[name].to_pylist()):
            if value is None or not value.strip() or value == "unclassified":
                reasons[index].append("missing_" + name)
    duration, size = numbers(table, "duration_seconds"), numbers(table, "video_size_bytes")
    for mask, reason in [(~np.isfinite(duration) | (duration <= 0), "invalid_duration"),
                         (~np.isfinite(size) | (size < 0), "invalid_size")]:
        for index in np.flatnonzero(mask):
            reasons[index].append(reason)
    keep = np.array([not reason for reason in reasons], dtype=bool)
    excluded = put(table.filter(~keep), "exclusion_reasons",
                   pa.array([reason for reason in reasons if reason], type=pa.list_(pa.string())))
    return table.filter(keep), excluded


class Data:
    def __init__(self, table, config):
        self.table = table
        self.n = len(table)
        self.h = numbers(table, "duration_seconds") / 3600
        self.b = numbers(table, "video_size_bytes") / 2**30
        self.keys = [table[field].to_pylist() for field in DIMENSIONS.values()]
        self.band, self.signature, self.penalty = measurements(table, config["technical_quality"])
        pixels = numbers(table, "pixels")
        computed = numbers(table, "width") * numbers(table, "height")
        pixels = np.where(np.isfinite(pixels), pixels, np.nan_to_num(computed, nan=0))
        uploaded = np.nan_to_num(numbers(table, "upload_timestamp"), nan=0)
        order = np.lexsort((np.arange(self.n), -uploaded, -pixels))
        self.rank = np.empty(self.n)
        self.rank[order] = (np.arange(self.n) + 1) / max(self.n, 1)

    def groups(self):
        groups = {}
        for index in range(self.n):
            key = tuple(column[index] for column in self.keys) + (int(self.band[index]), int(self.signature[index]))
            groups.setdefault(key, []).append(index)
        return [(key, np.asarray(indices, dtype=np.int64)) for key, indices in groups.items()]
