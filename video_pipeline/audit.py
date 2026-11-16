"""Recompute constraints from exported rows, independently of solver variables."""

import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from .common import DIMENSIONS, distribution, write_json
from .report import render


def distance(dimension, identifier, hours, minimum, maximum, target):
    below = max(0., minimum - hours)
    above = max(0., hours - maximum) if maximum is not None else 0.
    return dict(dimension=dimension, id=identifier, hours=float(hours), min_hours=minimum,
                max_hours=maximum, below_hours=below, above_hours=above,
                gap_hours=below + above, share_of_target=hours / target,
                passed=below + above <= 1e-7)


def technical(connection, config):
    names = {row[0]: row[1] for row in connection.execute("DESCRIBE final").fetchall()}

    def number(name):
        kinds = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "UTINYINT", "USMALLINT",
                 "UINTEGER", "UBIGINT", "FLOAT", "DOUBLE", "DECIMAL")
        return f'cast("{name}" AS DOUBLE)' if names.get(name, "").startswith(kinds) else "NULL::DOUBLE"

    def known(value):
        return f"({value} IS NOT NULL AND isfinite({value}) AND {value}>0)"

    quality = config["technical_quality"]
    width, height, fps, audio, duration = [number(name) for name in ["width", "height", "fps", "asr", "duration_seconds"]]
    side = f"least({width},{height})"
    side_known = f"({known(width)} AND {known(height)})"
    audio_known = known(audio)
    audio_pass = f"{audio}>={quality['min_audio_sample_rate_hz']}"
    if "has_audio" in names:
        audio_known += " AND has_audio IS NOT NULL" if names["has_audio"] == "BOOLEAN" else " AND false"
        audio_pass += " AND has_audio IS TRUE" if names["has_audio"] == "BOOLEAN" else " AND false"
    gates = [
        ("short_side", side_known, f"{side}>={quality['min_short_side']}", f"短边 ≥{quality['min_short_side']} px"),
        ("fps", known(fps), f"{fps}>={quality['min_fps']}", f"帧率 ≥{quality['min_fps']} fps"),
        ("duration", known(duration), f"{duration}>={quality['min_duration_seconds']}", f"时长 ≥{quality['min_duration_seconds']} 秒"),
        ("audio_sample_rate", audio_known, audio_pass, f"音频采样率 ≥{quality['min_audio_sample_rate_hz']} Hz"),
    ]
    count, hours = connection.execute("SELECT count(*), coalesce(sum(duration_seconds)/3600.,0) FROM final").fetchone()
    rows = []

    def calculate(key, known_expression, condition, label, share=None):
        good = f"coalesce(({known_expression}) AND ({condition}),false)"
        bad, bad_hours, unknown = connection.execute(
            f"SELECT count(*) FILTER(WHERE NOT {good}), "
            f"coalesce(sum(duration_seconds) FILTER(WHERE NOT {good})/3600.,0), "
            f"count(*) FILTER(WHERE NOT coalesce({known_expression},false)) FROM final"
        ).fetchone()
        qualified = hours - bad_hours
        minimum = config["target_total_hours"] * share if share is not None else None
        rows.append(dict(requirement=key, label=label, videos=count, failed_videos=bad,
                         failed_hours=float(bad_hours), failed_video_ratio=bad / count if count else 0,
                         unknown_videos=unknown, qualified_hours=float(qualified),
                         required_min_share=share, min_hours=minimum,
                         passed=qualified + 1e-7 >= minimum if minimum is not None else bad == 0))

    for key, known_expression, condition, label in gates:
        calculate(key, known_expression, condition, label)
    for rule in quality["resolution_shares"]:
        threshold = rule["min_short_side"]
        calculate(f"resolution_{threshold}", side_known, f"{side}>={threshold}",
                  f"短边 ≥{threshold} px 的小时占比", rule["min_share"])
    return dict(passed=all(row["passed"] for row in rows), checks=rows)


def audit(run):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    report = json.loads((run / "report.json").read_text())
    target, cap = config["target_total_hours"], config["max_gib_per_hour"]
    if pq.read_schema(run / "selected.parquet") != pq.read_schema(run / "candidates.parquet"):
        raise ValueError("Selected metadata schema differs from candidate schema")
    with duckdb.connect() as connection:
        connection.from_parquet(str(run / "selected.parquet")).create_view("final")
        connection.from_parquet(str(run / "candidates.parquet")).create_view("candidates")
        count, unique = connection.execute("SELECT count(*), count(DISTINCT source_uri) FROM final").fetchone()
        if count != unique or count != report["selected_rows"]:
            raise ValueError("Selected identities or row count differ")
        changed = connection.execute(
            "SELECT count(*) FROM (SELECT * FROM final EXCEPT ALL SELECT * FROM candidates)"
        ).fetchone()[0]
        if changed:
            raise ValueError("Selected rows contain changed or unknown metadata")
        expected = [json.loads(row[0]) for row in connection.execute("SELECT to_json(f) FROM final f").fetchall()]
        if json.loads((run / "selected.json").read_text()) != expected:
            raise ValueError("Selected JSON differs from selected Parquet")
        hours, gib = connection.execute(
            "SELECT coalesce(sum(duration_seconds)/3600.,0), coalesce(sum(video_size_bytes)/1073741824.,0) FROM final"
        ).fetchone()
        tolerance = config["total_tolerance"]
        quotas = [distance("total", "total", hours, target * (1 - tolerance), target * (1 + tolerance), target)]
        densities = [dict(dimension="total", id="total", hours=hours, gib=gib)]
        labels = {}
        for dimension, column in DIMENSIONS.items():
            values = {key: (amount, size) for key, amount, size in connection.execute(
                f"SELECT {column},sum(duration_seconds)/3600.,sum(video_size_bytes)/1073741824. FROM final GROUP BY 1"
            ).fetchall()}
            if dimension != "language":
                densities.extend(dict(dimension=dimension, id=key, hours=amount, gib=size)
                                 for key, (amount, size) in values.items() if key is not None)
            for rule in config["rules"][dimension]:
                labels[rule["id"]] = rule.get("name", rule["id"])
                amount, _ = values.get(rule["id"], (0., 0.))
                maximum = rule.get("max_share")
                quotas.append(distance(dimension, rule["id"], amount, target * rule.get("min_share", 0),
                                       target * maximum if maximum is not None else None, target))
        for row in densities:
            row["gib_per_hour"] = row["gib"] / row["hours"] if row["hours"] else None
            row["passed"] = row["gib"] <= cap * row["hours"] + 1e-7
        if not all(row["passed"] for row in densities):
            raise ValueError("Selected total or category average density exceeds its limit")
        quality = technical(connection, config)
    quota_passed = all(row["passed"] for row in quotas)
    result = dict(rows=count, hours=float(hours), hard_constraints_passed=True,
                  quota_requirements_passed=quota_passed, technical_quality_passed=quality["passed"],
                  passed=quota_passed and quality["passed"], density_checks=densities)
    write_json(run / "independent-audit.json", result)
    write_json(run / "quota-summary.json", dict(passed=quota_passed, checks=quotas))
    write_json(run / "technical-quality.json", quality)
    report.update(result)
    write_json(run / "report.json", report)
    stats = distribution(run / "selected.parquet")
    write_json(run / "hours-summary.json", stats)
    render(run / "statistics.html", "均衡重采样", report, stats, labels=labels,
           quotas=quotas, quality=quality["checks"], densities=densities, cap=cap)
    return result
