"""Build centers for every node in an external category tree."""

import hashlib
import json
from pathlib import Path

EMBEDDING = {
    "model_name": "Qwen3-Embedding-4B",
    "dimensions": 2560,
    "pooling": "last_token_left_padding",
    "max_length": 512,
    "center_max_length": 96,
    "normalization": "NFC_whitespace_2048chars_L2",
}


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def centers(config):
    result = []
    seen = set()
    for primary in config["primary"]:
        def add(node, path, secondary=None):
            identifier = node["id"]
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError("Category IDs must be unique nonempty strings")
            if not isinstance(node["name"], str) or not node["name"].strip():
                raise ValueError("Category names must be nonempty strings")
            seen.add(identifier)
            result.append(dict(id=identifier, name=node["name"], primary_id=primary["id"],
                               secondary_id=secondary, level=len(path), path=path, text=" / ".join(path)))

        add(primary, [primary["name"]])
        for secondary in primary.get("secondaries", []):
            path = [primary["name"], secondary["name"]]
            add(secondary, path, secondary["id"])
            for leaf in secondary.get("children", []):
                add(leaf, path + [leaf["name"]], secondary["id"])
    if len(result) < 2:
        raise ValueError("Provide at least two content centers")
    return result


def load(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("embedding", EMBEDDING) != EMBEDDING:
        raise ValueError("Use Qwen3-Embedding-4B, 2560 dimensions and 512/96 tokens")
    config["embedding"] = dict(EMBEDDING)
    centers(config)
    return config
