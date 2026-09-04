"""CLI for direct end-to-end teacher-KLD Shapley analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .kld_shapley_native import analyze
from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=50_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260966)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = analyze(
        args.design,
        args.run_root,
        run_prefix=args.run_prefix,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "selected_upgrade_layers": payload["selected_upgrade_layers"]}, sort_keys=True))


if __name__ == "__main__":
    main()
