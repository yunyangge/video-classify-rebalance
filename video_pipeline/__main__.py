"""Command-line entry points for classification and balanced selection."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Classify and rebalance one video metadata collection")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["centers", "classify"]:
        child = commands.add_parser(name)
        child.add_argument("--config", required=True)
        child.add_argument("--model", required=True)
        child.add_argument("--output", required=True)
        child.add_argument("--device", default="cuda:0")
        child.add_argument("--batch-size", type=int, default=128)
        child.add_argument("--max-padded-tokens", type=int, default=16384)
        if name == "classify":
            child.add_argument("--input", required=True)
            child.add_argument("--centers", required=True)
            child.add_argument("--forms", required=True)
            child.add_argument("--input-batch-size", type=int, default=8192)
            child.add_argument("--validate-only", action="store_true")
    child = commands.add_parser("rebalance")
    child.add_argument("--input", required=True)
    child.add_argument("--config", required=True)
    child.add_argument("--output", required=True)
    child.add_argument("--seed", help="Optional Parquet containing source_uri values")
    child.add_argument("--validate-only", action="store_true")
    child = commands.add_parser("audit")
    child.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.command == "centers":
        from .classify import embed_centers
        result = embed_centers(args)
    elif args.command == "classify":
        from .classify import run
        result = run(args)
    elif args.command == "rebalance":
        from .rebalance import run
        result = run(args)
    else:
        from .audit import audit
        result = audit(args.run)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
