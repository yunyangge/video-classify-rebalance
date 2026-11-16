"""Offline charts backed by the same JSON values as the output files."""

import json
from pathlib import Path


def render(path, title, report, stats, labels=None, quotas=None, quality=None, densities=None, cap=None):
    data = dict(title=title, report=report, stats=stats, labels=labels or {}, quotas=quotas or [],
                quality=quality or [], densities=densities or [], cap=cap)
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False)
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
    Path(path).write_text(template.replace("__DATA__", encoded), encoding="utf-8")
