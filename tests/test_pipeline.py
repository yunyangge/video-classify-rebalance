import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from video_pipeline import classify
from video_pipeline.audit import audit
from video_pipeline.common import distribution, write_json
from video_pipeline.data import Data, prepare, read_config
from video_pipeline.embeddings import Matcher, joined_evidence
from video_pipeline.rebalance import run
from video_pipeline.report import render
from video_pipeline.selection import select
from video_pipeline.taxonomy import EMBEDDING, centers, digest


def config(target=2):
    return dict(target_total_hours=target, total_tolerance=0, max_gib_per_hour=2.5,
                rules={key: [] for key in ["primary", "secondary", "production_form", "language"]},
                technical_quality=dict(min_short_side=100, min_fps=20, min_duration_seconds=10,
                                       min_audio_sample_rate_hz=40000, resolution_shares=[]),
                lp_seconds=2, solve_seconds=5, max_options_per_group=32, refinement_rounds=1)


def record(index, density=1, **changes):
    row = dict(source_uri=f"https://example.invalid/{index}", sample_id=str(index),
               source="generated", tag=["example"], primary_id="topic", secondary_id=None,
               production_form_id="form", language="language", duration_seconds=3600.,
               video_size_bytes=int(density * 2**30), width=300, height=300,
               fps=24., asr=48000., upload_timestamp=100 - index)
    return row | changes


class FakeModel:
    def __init__(self, direct=False):
        self.cfg = {"embedding": EMBEDDING}
        self.nodes = [dict(id="p", name="Primary", primary_id="p", secondary_id=None, level=1, path=["Primary"]),
                      dict(id="s", name="Secondary", primary_id="p", secondary_id="s", level=2, path=["Primary", "Secondary"])]
        self.by_id = {row["id"]: row for row in self.nodes}
        self.forms = [dict(id="form", name="Form")]
        self.calls, self.direct = [], direct

    def predict(self, texts):
        self.calls.append(texts)
        n = len(texts)
        return np.full(n, 1 if self.direct else 2), np.full(n, .8), np.full(n, .2), np.zeros(n, int), np.full(n, .7), np.full(n, .1)


class ClassificationTests(unittest.TestCase):
    def test_clean_joined_query_once_per_video(self):
        table = pa.table({"tag": [[" CAT! ", "play", "CAT!"], ["CAT!", "play"], []], "extra": [1, 2, 3]})
        model = FakeModel()
        result = classify.classify_table(table, model)
        self.assertEqual(model.calls, [["CAT!, play", "CAT!, play"]])
        self.assertEqual(result["extra"].to_pylist(), [1, 2, 3])
        self.assertEqual(result["content_node_id"].to_pylist(), ["s", "s", "unclassified"])
        self.assertEqual(result["evidence_empty"].to_pylist(), [False, False, True])
        self.assertIsNone(result["match_score"][2].as_py())

    def test_primary_endpoint_and_aliases(self):
        result = classify.classify_table(pa.table({"id": ["one"], "url": ["url"], "tags": [["text"]]}), FakeModel(True))
        self.assertEqual(result["sample_id"][0].as_py(), "one")
        self.assertEqual(result["content_node_id"][0].as_py(), "p")
        self.assertIsNone(result["secondary_id"][0].as_py())

    def test_json_fields_are_restored(self):
        table = pa.table({"tag": ['["one"]', '"two, three"'], "nested": ['{"x":1}', '{"x":[2]}']})
        model = FakeModel()
        result = classify.classify_table(table, model, ["tag", "nested"])
        self.assertEqual(model.calls, [["one", "two, three"]])
        self.assertEqual(json.loads(result["nested"][1].as_py()), {"x": [2]})

    def test_text_limits_and_empty_tags(self):
        self.assertEqual(joined_evidence({"tags": ["https://example.invalid", "123", "unknown"]}), "")
        self.assertEqual(len(joined_evidence({"tags": ["x" * 3000]})), 2048)

    def test_all_nodes_are_centers(self):
        rows = centers({"primary": [{"id": "p", "name": "P", "secondaries": [
            {"id": "s", "name": "S", "children": [{"id": "t", "name": "T"}]}]}]})
        self.assertEqual([r["text"] for r in rows], ["P", "P / S", "P / S / T"])
        self.assertEqual([r["level"] for r in rows], [1, 2, 3])

    def test_duplicate_center_ids_rejected(self):
        with self.assertRaises(ValueError):
            centers({"primary": [{"id": "x", "name": "P"}, {"id": "x", "name": "Q"}]})

    def test_real_matcher_ties_and_changed_vector_file(self):
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            matrix = np.zeros((2, 2560), dtype=np.float16); matrix[:, 0] = 1
            path = root / "center_vectors.npy"; np.save(path, matrix)
            spec = {"centers": [{"id": "one"}, {"id": "two"}]}
            write_json(root / "center_vectors.json", dict(run_fingerprint=digest(spec), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            args = Namespace(run_dir=root, min_score=-1., min_margin=0.)
            model = Matcher(args, {"embedding": EMBEDDING}, spec)
            labels, scores, margins = model.match(matrix[:1])
            self.assertEqual(labels.tolist(), [1]); self.assertEqual(margins.tolist(), [0.])
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                Matcher(args, {"embedding": EMBEDDING}, spec)

    def test_center_files_and_real_model_preflight(self):
        class Encoder:
            def __init__(self, model, device, spec, *args):
                self.spec = spec

            def encode(self, texts):
                result = np.zeros((len(texts), 2560), np.float16)
                result[:, 0] = 1
                return result

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "taxonomy.json"
            write_json(cfg, {"primary": [{"id": "one", "name": "One"}, {"id": "two", "name": "Two"}]})
            forms = root / "forms.json"
            write_json(forms, {"forms": [{"id": "f1", "name": "One", "text": "First"},
                                         {"id": "f2", "name": "Two", "text": "Second"}]})
            args = Namespace(config=cfg, model=root, output=root / "centers", centers=root / "centers",
                             device="cpu", batch_size=128, max_padded_tokens=16384, forms=forms)
            with patch.object(classify, "Encoder", Encoder):
                self.assertEqual(classify.embed_centers(args)["dimensions"], 2560)
                model = classify.Model(args)
                self.assertIsNone(model.encoder)
                result = model.predict(["whole video text"])
                self.assertEqual(result[0].tolist(), [1])

    def test_classification_delivery_and_empty_input(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); forms = root / "forms.json"; forms.write_text("{}")
            for name, tags in [("normal", [["query"], []]), ("empty", [])]:
                source = root / f"{name}.parquet"
                pq.write_table(pa.table({"tag": pa.array(tags, type=pa.list_(pa.string()))}), source)
                args = Namespace(input=source, output=root / name, forms=forms, batch_size=128,
                                 input_batch_size=1, max_padded_tokens=16384, device="cpu", validate_only=False)
                with patch.object(classify, "Model", return_value=FakeModel()):
                    result = classify.run(args)
                self.assertEqual(result["rows"], len(tags))
                self.assertEqual(len(json.loads((root / name / "classified.json").read_text())), len(tags))
                self.assertTrue((root / name / "statistics.html").is_file())

    def test_distribution_ignores_nonfinite_hours(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rows.parquet"
            pq.write_table(pa.table({"duration_seconds": [3600., float("nan"), float("inf"), 0., -1.]}), path)
            stats = distribution(path)
            self.assertEqual(stats["hours"], 1)
            self.assertEqual(stats["missing_duration_rows"], 4)


class SelectionTests(unittest.TestCase):
    def select(self, rows, cfg=None, seed=None):
        cfg = cfg or config()
        data = Data(pa.Table.from_pylist(rows), cfg)
        indices, _ = select(data, cfg, np.zeros(len(rows), bool) if seed is None else np.asarray(seed))
        return indices.tolist()

    def test_high_and_low_density_can_pair(self):
        self.assertEqual(self.select([record(0, 4), record(1, 1)]), [0, 1])

    def test_density_applies_to_unconfigured_categories(self):
        rows = [record(0, 4, primary_id="high"), record(1, 1, primary_id="low")]
        self.assertEqual(self.select(rows), [1])

    def test_all_high_density_returns_empty(self):
        self.assertEqual(self.select([record(0, 4), record(1, 4)]), [])

    def test_total_hours_take_priority_over_technical_failures(self):
        self.assertEqual(self.select([record(0, fps=1), record(1)]), [0, 1])

    def test_technical_quality_precedes_density(self):
        self.assertEqual(self.select([record(0, 2, fps=1), record(1, 1)], config(1)), [1])

    def test_density_then_original_quality(self):
        self.assertEqual(self.select([record(0, 1), record(1, 2)], config(1)), [1])
        self.assertEqual(self.select([record(0, 1), record(1, 1, width=600)], config(1)), [1])

    def test_category_minimum_and_seed_can_change(self):
        cfg = config(1)
        cfg["rules"]["primary"] = [dict(id="required", min_share=1, max_share=1)]
        rows = [record(0, primary_id="other"), record(1, primary_id="required")]
        self.assertEqual(self.select(rows, cfg, [True, False]), [1])

    def test_resolution_uses_fixed_target(self):
        cfg = config(2)
        cfg["technical_quality"]["resolution_shares"] = [dict(min_short_side=1440, min_share=.5)]
        rows = [record(0), record(1, width=2560, height=1440), record(2, density=2)]
        result = self.select(rows, cfg)
        self.assertEqual(len(result), 2)
        self.assertIn(1, result)

    def test_large_group_generates_cut_near_target(self):
        indices = self.select([record(i) for i in range(12)], config(5))
        self.assertEqual(len(indices), 5)

    def test_random_small_pools_match_exhaustive_selection(self):
        import itertools
        import random
        for seed in range(12):
            randomizer = random.Random(seed)
            rows = [record(i, randomizer.choice([1, 2, 3, 4]),
                           primary_id=f"topic-{i % 2}", fps=randomizer.choice([10., 30.]),
                           height=randomizer.choice([200, 600]), width=600) for i in range(7)]
            cfg = config(3)
            cfg["rules"]["primary"] = [dict(id=f"topic-{i}", min_share=.5, max_share=.5) for i in range(2)]
            cfg["technical_quality"]["resolution_shares"] = [dict(min_short_side=500, min_share=.5)]
            density_order = sorted(range(7), key=lambda i: (-rows[i]["video_size_bytes"], -(rows[i]["width"] * rows[i]["height"]), -rows[i]["upload_timestamp"], i))
            quality_order = sorted(range(7), key=lambda i: (-(rows[i]["width"] * rows[i]["height"]), -rows[i]["upload_timestamp"], i))
            density_rank = {index: rank + 1 for rank, index in enumerate(density_order)}
            quality_rank = {index: rank + 1 for rank, index in enumerate(quality_order)}

            def score(indices):
                groups = [[i for i in indices if rows[i]["primary_id"] == f"topic-{topic}"] for topic in range(2)]
                for group in [indices] + groups:
                    if sum(rows[i]["video_size_bytes"] / 2**30 for i in group) > 2.5 * len(group) + 1e-7:
                        return None
                return (abs(len(indices) - 3), sum(abs(len(group) - 1.5) for group in groups),
                        sum(rows[i]["fps"] < 20 for i in indices),
                        max(0, 1.5 - sum(rows[i]["height"] >= 500 for i in indices)),
                        sum(density_rank[i] for i in indices), sum(quality_rank[i] for i in indices))

            best = min(value for mask in itertools.product([0, 1], repeat=7)
                       if (value := score([i for i, keep in enumerate(mask) if keep])) is not None)
            self.assertEqual(score(self.select(rows, cfg)), best, f"random seed {seed}")

    def test_isolated_solver_and_deadline_fallback(self):
        from scipy.sparse import csr_matrix
        from video_pipeline.mip import isolated
        values, _ = isolated(np.array([-1., 0.]), csr_matrix([[1., 1.]]),
                             np.array([1.]), np.array([1.]), np.array([0., 1.]), 5., 2)
        self.assertEqual(values.tolist(), [1., 0.])
        original = np.array([0., 1.])
        values, status = isolated(np.array([-1., 0.]), csr_matrix([[1., 1.]]),
                                  np.array([1.]), np.array([1.]), original, 0., 2)
        self.assertEqual(values.tolist(), original.tolist())
        self.assertIn("time_limit", status)

    def test_infeasible_seed_is_replaced(self):
        self.assertEqual(self.select([record(0, 4), record(1, 1)], config(1), [True, False]), [1])


class DeliveryTests(unittest.TestCase):
    def execute(self, root, rows=None, cfg=None):
        source, settings = root / "input.parquet", root / "rules.json"
        pq.write_table(pa.Table.from_pylist(rows or [record(0, 4), record(1, 1)]), source)
        write_json(settings, cfg or config())
        args = Namespace(input=source, config=settings, output=root / "run", seed=None, validate_only=False)
        return run(args), args.output

    def test_real_delivery_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as td:
            result, out = self.execute(Path(td))
            self.assertTrue(result["passed"])
            self.assertEqual(result["selected_rows"], 2)
            self.assertTrue(audit(out)["passed"])
            self.assertEqual(set(pq.read_table(out / "selected.parquet").column_names), set(record(0)))

    def test_metadata_tampering_detected(self):
        with tempfile.TemporaryDirectory() as td:
            _, out = self.execute(Path(td))
            table = pq.read_table(out / "selected.parquet")
            table = table.set_column(table.schema.get_field_index("tag"), "tag", pa.array([["changed"], ["changed"]]))
            pq.write_table(table, out / "selected.parquet")
            with self.assertRaisesRegex(ValueError, "changed"):
                audit(out)

    def test_json_tampering_detected(self):
        with tempfile.TemporaryDirectory() as td:
            _, out = self.execute(Path(td))
            rows = json.loads((out / "selected.json").read_text()); rows[0]["fps"] = 1
            write_json(out / "selected.json", rows)
            with self.assertRaisesRegex(ValueError, "JSON"):
                audit(out)

    def test_all_excluded_still_outputs_schema(self):
        with tempfile.TemporaryDirectory() as td:
            result, out = self.execute(Path(td), [record(0, primary_id="unclassified")])
            self.assertEqual(result["selected_rows"], 0)
            self.assertEqual(result["excluded_rows"], 1)
            self.assertFalse(result["passed"])
            self.assertEqual(json.loads((out / "selected.json").read_text()), [])

    def test_duplicate_identities_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / "rows.parquet"
            pq.write_table(pa.Table.from_pylist([record(0), record(0)]), source)
            with self.assertRaisesRegex(ValueError, "unique"):
                prepare(source)

    def test_invalid_config_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            for value in [float("nan"), -1, True]:
                changed = config(); changed["target_total_hours"] = value
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    read_config(path)

    def test_report_escapes_external_labels(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "report.html"
            render(path, "test", {}, dict(videos=0, hours=0, dimensions={}), labels={"x": "</script><script>bad()</script>"})
            self.assertNotIn("</script><script>bad()", path.read_text())


if __name__ == "__main__":
    unittest.main()
