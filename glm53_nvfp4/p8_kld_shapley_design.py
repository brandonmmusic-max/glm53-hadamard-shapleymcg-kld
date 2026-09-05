"""CLI for freezing a direct-KLD native P8 Shapley design."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .kld_shapley_native import ROUTED_LAYERS, build_design
from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--fit-roles", type=Path)
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--capture-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=20260966)
    parser.add_argument("--layers", nargs="+", type=int, default=list(ROUTED_LAYERS))
    parser.add_argument("--base-bpw", type=float, default=4.25)
    parser.add_argument("--upgrade-bpw", type=float, default=8.25)
    parser.add_argument("--budget-bpw", type=float, default=6.0)
    parser.add_argument("--game", default="p8-k4-to-scalar-mxfp8")
    parser.add_argument("--upgrade-slots", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    roles = json.loads(args.roles.read_text())
    if roles.get("roles", {}).get("selection") or roles.get("roles", {}).get("confirmation") or roles.get("roles", {}).get("final"):
        raise RuntimeError("adaptive Shapley roles file exposes a protected role")
    design = build_design(
        args.seed,
        layers=args.layers,
        base_bpw=args.base_bpw,
        upgrade_bpw=args.upgrade_bpw,
        budget_bpw=args.budget_bpw,
        game=args.game,
        upgrade_slots_override=args.upgrade_slots,
    )
    design["roles"] = str(args.roles.resolve())
    design["roles_sha256"] = sha256_file(args.roles)
    if args.game == "p8-k4-to-scalar-mxfp8":
        required = {
            "fit_roles": args.fit_roles,
            "source_index": args.source_index,
            "capture_manifest": args.capture_manifest,
        }
        missing = [name for name, path in required.items() if path is None]
        if missing:
            raise RuntimeError(
                "native physical design requires " + ", ".join(sorted(missing))
            )
        fit_roles = json.loads(args.fit_roles.read_text())
        if not fit_roles.get("roles", {}).get("fit"):
            raise RuntimeError("encoder fit roles are empty")
        design["encoder_inputs"] = {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in required.items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(design, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "unique_coalitions": design["unique_coalitions"], "upgrade_slots": design["allocation"]["upgrade_slots"]}, sort_keys=True))


if __name__ == "__main__":
    main()
