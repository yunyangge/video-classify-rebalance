"""File and Arrow helpers shared by the two stages."""

import json
from pathlib import Path

import duckdb

DIMENSIONS = {
    "primary": "primary_id",
    "secondary": "secondary_id",
    "production_form": "production_form_id",
    "language": "language",
}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def put(table, name, values):
    index = table.schema.get_field_index(name)
    if index < 0:
        return table.append_column(name, values)
    return table.set_column(index, name, values)


def export_json(parquet_path, output_path):
    with duckdb.connect() as connection:
        connection.from_parquet(str(parquet_path)).create_view("records")
        connection.execute("COPY records TO ? (FORMAT JSON, ARRAY true)", [str(output_path)])


def distribution(parquet_path):
    with duckdb.connect() as connection:
        connection.from_parquet(str(parquet_path)).create_view("records")
        names = {row[0] for row in connection.execute("DESCRIBE records").fetchall()}
        duration = "try_cast(duration_seconds AS DOUBLE)" if "duration_seconds" in names else "NULL::DOUBLE"
        valid = f"isfinite({duration}) AND {duration}>0"
        rows, hours, missing = connection.execute(
            f"SELECT count(*), coalesce(sum({duration}) FILTER(WHERE {valid}),0)/3600., "
            f"count(*) FILTER(WHERE NOT coalesce({valid},false)) FROM records"
        ).fetchone()
        dimensions = {}
        for dimension, field in DIMENSIONS.items():
            if field not in names:
                continue
            dimensions[dimension] = [
                dict(id=key, videos=count, hours=amount)
                for key, count, amount in connection.execute(
                    f"SELECT {field}, count(*), coalesce(sum({duration}) FILTER(WHERE {valid}),0)/3600. "
                    f"FROM records GROUP BY 1 ORDER BY 1"
                ).fetchall()
            ]
    return dict(videos=rows, hours=hours, missing_duration_rows=missing, dimensions=dimensions)
