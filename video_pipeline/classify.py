"""Classify one metadata table using a shared content/form query vector."""
import hashlib
import json
import time
from pathlib import Path
from argparse import Namespace

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .common import put, new_directory, write_json, export_json, distribution
from .embeddings import Encoder, Matcher, joined_evidence
from .taxonomy import load, centers, digest
from .report import render

PREDICTION_FIELDS = {
    "primary_id": pa.string(), "primary_name": pa.string(),
    "secondary_id": pa.string(), "secondary_name": pa.string(),
    "content_node_id": pa.string(), "content_node_name": pa.string(),
    "content_node_level": pa.int32(), "content_path": pa.string(),
    "match_score": pa.float32(), "match_margin": pa.float32(),
    "production_form_id": pa.string(), "production_form_name": pa.string(),
    "form_score": pa.float32(), "form_margin": pa.float32(), "evidence_empty": pa.bool_(),
}


def specification(config):
    return dict(taxonomy_fingerprint=digest(config), embedding=config["embedding"],
                centers=centers(config), query_unit="joined_tags_per_video")


def load_centers(config_path, folder):
    config = load(config_path)
    spec = specification(config)
    stored = json.loads((Path(folder) / "run.json").read_text())
    if stored != spec:
        raise ValueError("Centers were built with a different taxonomy; rebuild them")
    return config, spec


def embed_centers(args):
    config = load(args.config)
    spec = specification(config)
    folder = new_directory(args.output)
    encoder = Encoder(Path(args.model), args.device,
                      dict(config["embedding"], max_length=96),
                      args.batch_size, args.max_padded_tokens)
    matrix = encoder.encode([node["text"] for node in spec["centers"]])
    path = folder / "center_vectors.npy"
    np.save(path, matrix)
    write_json(folder / "run.json", spec)
    write_json(folder / "center_vectors.json", dict(
        run_fingerprint=digest(spec), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(centers=len(matrix), dimensions=matrix.shape[1])


def canonical(table):
    names = set(table.column_names)
    for (target, alias, typ) in [('sample_id', 'id', pa.string()), ('source_uri', 'url', pa.string()), ('source', 'platform', pa.string()), ('tag', 'tags', pa.list_(pa.string()))]:
        if target not in names:
            value = table[alias] if alias in names else pa.nulls(len(table), type=typ)
            table = put(table, target, value)
    return table

class Model:

    def __init__(self, args):
        self.cfg, self.spec = load_centers(args.config, args.centers)
        e = self.cfg['embedding']
        if e['model_name'] != 'Qwen3-Embedding-4B' or e['dimensions'] != 2560 or e['max_length'] != 512 or (e['center_max_length'] != 96):
            raise ValueError('Requires Qwen3-Embedding-4B, full 2560 dimensions, 512/96 tokens')
        self.nodes = self.spec['centers']
        self.by_id = {n['id']: n for n in self.nodes}
        self.forms = json.loads(Path(args.forms).read_text())['forms']
        if len(self.forms) < 2 or any((not all((k in f for k in ['id', 'name', 'text'])) for f in self.forms)):
            raise ValueError('Invalid external form centers')
        self.matcher = Matcher(Namespace(run_dir=Path(args.centers), min_score=-1.0, min_margin=0.0), self.cfg, self.spec)
        self.encoder = None
        self.args = args

    def load(self):
        if self.encoder is not None:
            return
        self.encoder = Encoder(Path(self.args.model), self.args.device, self.cfg['embedding'], self.args.batch_size, self.args.max_padded_tokens)
        self.encoder.spec = dict(self.encoder.spec, max_length=96)
        self.form_vectors = self.encoder.encode([f['text'] for f in self.forms]).astype(np.float32)
        norms = np.linalg.norm(self.form_vectors, axis=1, keepdims=True)
        if not np.isfinite(self.form_vectors).all() or np.any(norms == 0):
            raise ValueError('Invalid form center vectors')
        self.form_vectors /= norms
        self.encoder.spec = dict(self.encoder.spec, max_length=512)

    def predict(self, texts):
        self.load()
        x = self.encoder.encode(texts)
        (labels, scores, margins) = self.matcher.match(x)
        fx = x.astype(np.float32)
        norms = np.linalg.norm(fx, axis=1)
        if not np.isfinite(fx).all() or np.any(norms == 0):
            raise ValueError('Invalid query vectors')
        fx /= norms[:, None]
        sim = np.clip(fx @ self.form_vectors.T, -1, 1)
        fi = sim.argmax(1)
        fs = sim[np.arange(len(fi)), fi]
        fm = fs - np.partition(sim, -2, axis=1)[:, -2]
        return (labels, scores, margins, fi, fs, fm)

def classify_table(table, model, json_columns=()):
    for name in json_columns:
        if name in table.column_names:
            table = put(table, name, pa.array(table[name].to_pylist(), type=pa.json_()))
    table = canonical(table)
    tags = table['tag'].to_pylist()
    if getattr(table.schema.field('tag').type, 'extension_name', None) == 'arrow.json':
        tags = [json.loads(x) if x is not None else None for x in tags]
    tags = [x if x is None or isinstance(x, (str, list, tuple)) else [] for x in tags]
    texts = [joined_evidence({'tags': t}, 'tags') for t in tags]
    ix = [i for (i, t) in enumerate(texts) if t]
    result = {k: [None] * len(table) for k in PREDICTION_FIELDS}
    result['evidence_empty'] = [not bool(t) for t in texts]
    for i in range(len(table)):
        for k in ['primary_id', 'content_node_id', 'production_form_id']:
            result[k][i] = 'unclassified'
    if ix:
        (labels, score, margin, fi, fs, fm) = model.predict([texts[i] for i in ix])
        for (j, i) in enumerate(ix):
            node = model.nodes[int(labels[j]) - 1] if labels[j] else None
            if node:
                pid = node['primary_id']
                sid = node.get('secondary_id')
                for (k, v) in {'primary_id': pid, 'primary_name': model.by_id[pid]['name'], 'secondary_id': sid, 'secondary_name': model.by_id[sid]['name'] if sid else None, 'content_node_id': node['id'], 'content_node_name': node['name'], 'content_node_level': node['level'], 'content_path': ' / '.join(node['path'])}.items():
                    result[k][i] = v
            result['match_score'][i] = float(score[j])
            result['match_margin'][i] = float(margin[j])
            form = model.forms[int(fi[j])]
            for (k, v) in {'production_form_id': form['id'], 'production_form_name': form['name'], 'form_score': float(fs[j]), 'form_margin': float(fm[j])}.items():
                result[k][i] = v
    for (k, typ) in PREDICTION_FIELDS.items():
        table = put(table, k, pa.array(result[k], type=typ))
    return table


def run(args):
    path = Path(args.input)
    schema = pq.ParquetFile(path).schema_arrow
    if set(schema.names) & set(PREDICTION_FIELDS):
        raise ValueError("Input already contains classification fields")
    if args.batch_size <= 0 or args.input_batch_size <= 0 or args.max_padded_tokens < 512:
        raise ValueError("Use positive batch sizes and a padded-token budget of at least 512")
    model = Model(args)
    if args.validate_only:
        return dict(validated=True, centers=len(model.nodes), dimensions=2560)
    out = new_directory(args.output)
    started = time.perf_counter()
    writer = None
    rows = 0
    with duckdb.connect() as connection:
        connection.from_parquet(str(path)).create_view("input")
        json_columns = [row[0] for row in connection.execute("DESCRIBE input").fetchall() if row[1] == "JSON"]
        try:
            stream = connection.execute("SELECT * FROM input").to_arrow_reader(args.input_batch_size)
            for batch in stream:
                table = classify_table(pa.Table.from_batches([batch]), model, json_columns)
                if writer is None:
                    writer = pq.ParquetWriter(out / "classified.partial.parquet", table.schema, compression="zstd")
                writer.write_table(table)
                rows += len(table)
            if writer is None:
                table = classify_table(pa.Table.from_batches([], schema=schema), model)
                writer = pq.ParquetWriter(out / "classified.partial.parquet", table.schema, compression="zstd")
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
    (out / "classified.partial.parquet").rename(out / "classified.parquet")
    export_json(out / "classified.parquet", out / "classified.json")
    stats = distribution(out / "classified.parquet")
    write_json(out / "hours-summary.json", stats)
    report = dict(completed=True, rows=rows, seconds=time.perf_counter() - started,
                  model="Qwen3-Embedding-4B", dimensions=2560, query_tokens=512,
                  center_tokens=96, taxonomy_fingerprint=digest(model.cfg),
                  form_spec_sha256=hashlib.sha256(Path(args.forms).read_bytes()).hexdigest(),
                  device=args.device, batch_size=args.batch_size,
                  query_unit="joined_tags_per_video")
    write_json(out / "report.json", report)
    labels = {node["id"]: node["name"] for node in model.nodes}
    labels.update({form["id"]: form["name"] for form in model.forms})
    render(out / "statistics.html", "分类统计", report, stats, labels=labels)
    return report
