"""CPU-only gates and receipts for the all-42-layer coupled P8 sidecar build.

The build orchestration (``scripts/build_full_coupled_p8_all42.sh``) calls these
subcommands between the existing encoder, packer and postwrite verifier.  Nothing
here touches a GPU, opens a protected role, or restores production.

Subcommands
-----------
guard
    Fail closed unless production is inactive, port 8000 is unbound, every
    requested GPU is idle, the filesystem has the requested free bytes, and the
    campaign's apparent new bytes plus the requested reservation stay under the
    owner-approved ceiling.
reuse-layer
    Hard-link (or copy) an already encoded layer's four rank sidecars into the
    flat runtime sidecar directory after re-hashing them against the source
    packer receipt, and record the provenance.
layer-complete
    Verify one layer's four rank files against the packer receipt, the postwrite
    receipt, the coupled rank schema, the exact 4.25 weight-payload rate and an
    explicit allowlist of encoder design hashes.
manifest
    Write the pinned full-checkpoint manifest for layers 3..44 with exact byte
    accounting (weight payload, coupled metadata, headers) per rank and in total.
ledger
    Append one apparent-bytes snapshot line for the storage ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LAYERS = tuple(range(3, 45))
WORLD_SIZE = 4
COUPLED_RANK_SCHEMA = "glm53-p8-coupled-h512-h128-tp4-rank.v1"
COUPLED_SIDECAR_SCHEMA = "glm53-p8-mcg-coupled-scale-tp4-sidecars.v2"
COUPLED_BOUNDARY = "coupled-h512-h128-suh-svh-v1"
POSTWRITE_SCHEMA_PREFIX = "glm53-p8-coupled"
MANIFEST_SCHEMA = "glm53-p8-full-coupled-all42-tp4-checkpoint.v1"
WEIGHT_TENSORS = ("w13_trellis", "w2_trellis", "w13_scale_ue8m0", "w2_scale_ue8m0")
METADATA_TENSORS = (
    "gate_up_suh_fp16",
    "down_svh_fp16",
    "intermediate_scales_fp16",
    "coupled_sign_draw_u8",
)
PRODUCTION_SHAPES = {
    "w13_trellis": [2, 288, 256, 32, 64],
    "w2_trellis": [288, 32, 256, 64],
    "w13_scale_ue8m0": [288, 1024, 128],
    "w2_scale_ue8m0": [288, 4096, 16],
    "gate_up_suh_fp16": [4096],
    "intermediate_scales_fp16": [288, 1536],
    "down_svh_fp16": [4096],
    "coupled_sign_draw_u8": [288],
}
PRODUCTION_RANK_BYTES = 963_497_456
GPU_IDLE_MIB = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safetensors_header(path: Path) -> tuple[dict[str, str], dict[str, dict], int]:
    """Return (metadata, tensors, header_bytes) without importing torch."""
    with Path(path).open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"truncated safetensors prefix: {path}")
        (length,) = struct.unpack("<Q", prefix)
        header = json.loads(handle.read(length))
    metadata = header.pop("__metadata__", {}) or {}
    return metadata, header, 8 + length


def tree_apparent_bytes(path: Path, seen: set[tuple[int, int]] | None = None) -> int:
    """Apparent bytes under ``path``; a hard-linked file is charged once per ``seen`` set.

    Checkpoints reuse layers by hard link (pilot -> full build -> mixed-rate assembly), so
    callers that charge several campaign paths against one ceiling pass a shared ``seen``
    set and the linked bytes count once; a fresh set charges every link.
    """
    path = Path(path)
    if not path.exists():
        return 0
    if seen is None:
        seen = set()

    def charge(st: os.stat_result) -> int:
        if st.st_nlink > 1:
            key = (st.st_dev, st.st_ino)
            if key in seen:
                return 0
            seen.add(key)
        return st.st_size

    if path.is_file():
        return charge(path.lstat())
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                else:
                    total += charge(entry.stat(follow_symlinks=False))
    return total


def paths_apparent_bytes(paths) -> dict[str, int]:
    """Per-path apparent bytes with hard links charged once across all paths (first path wins)."""
    seen: set[tuple[int, int]] = set()
    return {str(path): tree_apparent_bytes(Path(path), seen) for path in paths}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- gates


def production_off(run=None) -> dict:
    run = run or subprocess.run  # late binding so tests can patch subprocess.run
    states = {}
    for unit, scope in (("klc-backend.service", "user"), ("klc-model-stack.timer", "system")):
        command = (
            ["systemctl", "--user", "is-active", unit]
            if scope == "user"
            else ["systemctl", "is-active", unit]
        )
        result = run(command, text=True, capture_output=True)
        state = result.stdout.strip()
        if state != "inactive":
            raise RuntimeError(f"{scope} production unit must remain inactive: {unit}={state!r}")
        states[unit] = state
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", 8000))
        except OSError as error:
            raise RuntimeError(f"port 8000 must stay unbound: {error}") from error
    return {"units": states, "port_8000": "unbound", "restore_policy": "never"}


def gpu_idle(gpus: list[int], run=None) -> dict:
    run = run or subprocess.run
    apps = run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True,
        capture_output=True,
    )
    if apps.returncode != 0:
        raise RuntimeError(f"nvidia-smi compute-apps query failed: {apps.stderr.strip()}")
    running = [line for line in apps.stdout.splitlines() if line.strip()]
    if running:
        raise RuntimeError(f"GPU compute processes present: {running}")
    memory = run(
        ["nvidia-smi", "--query-gpu=index,memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
    )
    if memory.returncode != 0:
        raise RuntimeError(f"nvidia-smi memory query failed: {memory.stderr.strip()}")
    rows = {}
    for line in memory.stdout.splitlines():
        if not line.strip():
            continue
        index, used, temperature = (part.strip() for part in line.split(","))
        rows[int(index)] = {"memory_used_mib": int(used), "temperature_c": int(temperature)}
    for gpu in gpus:
        if gpu not in rows:
            raise RuntimeError(f"GPU {gpu} not reported by nvidia-smi")
        if rows[gpu]["memory_used_mib"] >= GPU_IDLE_MIB:
            raise RuntimeError(f"GPU {gpu} is not idle: {rows[gpu]}")
    return {gpu: rows[gpu] for gpu in gpus}


def cmd_guard(args: argparse.Namespace) -> dict:
    result = {"checked_at": utc_now(), "status": "fail"}
    result["production"] = production_off()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if gpus:
        result["gpus"] = gpu_idle(gpus)
    campaign_bytes = sum(paths_apparent_bytes(args.campaign_path).values())
    free_bytes = shutil.disk_usage(args.root).free
    result.update(
        {
            "campaign_apparent_bytes": campaign_bytes,
            "reservation_bytes": args.need_bytes,
            "max_new_bytes": args.max_new_bytes,
            "filesystem_free_bytes": free_bytes,
            "filesystem_min_free_bytes": args.min_free_bytes,
        }
    )
    if campaign_bytes + args.need_bytes > args.max_new_bytes:
        raise RuntimeError(
            f"campaign bytes {campaign_bytes} + reservation {args.need_bytes} exceed ceiling {args.max_new_bytes}"
        )
    if free_bytes < args.need_bytes + args.min_free_bytes:
        raise RuntimeError(
            f"filesystem free {free_bytes} < reservation {args.need_bytes} + margin {args.min_free_bytes}"
        )
    result["status"] = "pass"
    return result


# ------------------------------------------------------------------- rank checks


def inspect_rank(
    path: Path,
    *,
    layer: int,
    rank: int,
    allowed_design_sha256: set[str],
    strict_geometry: bool = True,
) -> dict:
    metadata, tensors, header_bytes = safetensors_header(path)
    required = {
        "schema": COUPLED_RANK_SCHEMA,
        "layer": str(layer),
        "rank": str(rank),
        "world_size": str(WORLD_SIZE),
        "bits": "4",
        "alphabet": "e4m3",
        "scale": "ue8m0-k32",
        "law": "procedural-mcg-alpha2",
        "boundary": COUPLED_BOUNDARY,
        "full_coupled": "true",
        "ldlq": "false",
        "weight_payload_bpw": "4.25",
    }
    mismatched = {key: metadata.get(key) for key, value in required.items() if metadata.get(key) != value}
    if mismatched:
        raise RuntimeError(f"rank metadata mismatch {path}: {mismatched}")
    design = metadata.get("source_design_sha256", "")
    if design not in allowed_design_sha256:
        raise RuntimeError(f"rank {path} carries design {design!r} outside the allowlist")
    names = set(tensors)
    expected_names = set(WEIGHT_TENSORS) | set(METADATA_TENSORS)
    if names != expected_names:
        raise RuntimeError(f"rank {path} tensor set differs: {sorted(names ^ expected_names)}")
    if strict_geometry:
        for name, shape in PRODUCTION_SHAPES.items():
            if list(tensors[name]["shape"]) != shape:
                raise RuntimeError(f"rank {path} tensor {name} shape {tensors[name]['shape']} != {shape}")

    def nbytes(name: str) -> int:
        start, stop = tensors[name]["data_offsets"]
        return int(stop) - int(start)

    def numel(name: str) -> int:
        count = 1
        for dim in tensors[name]["shape"]:
            count *= int(dim)
        return count

    weight_bytes = sum(nbytes(name) for name in WEIGHT_TENSORS)
    metadata_bytes = sum(nbytes(name) for name in METADATA_TENSORS)
    logical = 32 * (numel("w13_scale_ue8m0") + numel("w2_scale_ue8m0"))
    if weight_bytes * 8 != 17 * logical // 4 or (17 * logical) % 4:
        raise RuntimeError(
            f"rank {path} weight payload {weight_bytes} bytes is not exactly 4.25 bpw for {logical} weights"
        )
    file_bytes = path.stat().st_size
    if file_bytes != header_bytes + weight_bytes + metadata_bytes:
        raise RuntimeError(f"rank {path} file size {file_bytes} != header + tensors")
    if strict_geometry and file_bytes != PRODUCTION_RANK_BYTES:
        raise RuntimeError(f"rank {path} file size {file_bytes} != {PRODUCTION_RANK_BYTES}")
    return {
        "path": str(path.resolve()),
        "rank": rank,
        "bytes": file_bytes,
        "header_bytes": header_bytes,
        "weight_payload_bytes": weight_bytes,
        "metadata_bytes": metadata_bytes,
        "logical_elements": logical,
        "source_design_sha256": design,
        "exl3_scale_source_sha256": metadata.get("exl3_scale_source_sha256"),
    }


def rank_path(sidecars: Path, layer: int, rank: int) -> Path:
    return Path(sidecars) / f"p8-layer-{layer:03d}-tp4-rank-{rank}.safetensors"


def load_packer_receipt(path: Path, layer: int) -> dict:
    receipt = json.loads(Path(path).read_text())
    if (
        receipt.get("schema") != COUPLED_SIDECAR_SCHEMA
        or receipt.get("layer") != layer
        or receipt.get("world_size") != WORLD_SIZE
        or receipt.get("boundary") != COUPLED_BOUNDARY
        or receipt.get("weight_payload_bpw") != 4.25
        or receipt.get("ldlq") is not False
    ):
        raise RuntimeError(f"packer receipt header differs for layer {layer}: {path}")
    ranks = receipt.get("ranks")
    if not isinstance(ranks, list) or [row.get("rank") for row in ranks] != list(range(WORLD_SIZE)):
        raise RuntimeError(f"packer receipt lacks ranks 0..3 exactly once: {path}")
    return receipt


def load_postwrite_receipt(path: Path, layer: int) -> dict:
    receipt = json.loads(Path(path).read_text())
    if (
        not str(receipt.get("schema", "")).startswith(POSTWRITE_SCHEMA_PREFIX)
        or receipt.get("layer") != layer
        or receipt.get("status") != "pass"
    ):
        raise RuntimeError(f"postwrite receipt is not a layer-{layer} pass: {path}")
    return receipt


def verify_layer(
    *,
    layer: int,
    sidecars: Path,
    packer_receipt: Path,
    postwrite_receipt: Path,
    allowed_design_sha256: set[str],
    strict_geometry: bool = True,
    rehash: bool = True,
) -> dict:
    packer = load_packer_receipt(packer_receipt, layer)
    postwrite = load_postwrite_receipt(postwrite_receipt, layer)
    rows = []
    designs = set()
    for entry in packer["ranks"]:
        rank = int(entry["rank"])
        path = rank_path(sidecars, layer, rank)
        if not path.is_file():
            raise RuntimeError(f"missing rank sidecar {path}")
        info = inspect_rank(
            path,
            layer=layer,
            rank=rank,
            allowed_design_sha256=allowed_design_sha256,
            strict_geometry=strict_geometry,
        )
        if info["bytes"] != int(entry["bytes"]):
            raise RuntimeError(f"rank {path} bytes differ from packer receipt")
        if rehash:
            digest = sha256_file(path)
            if digest != entry["sha256"]:
                raise RuntimeError(f"rank {path} sha256 differs from packer receipt")
            info["sha256"] = digest
        else:
            info["sha256"] = entry["sha256"]
        if info["source_design_sha256"] != entry.get("source_design_sha256"):
            raise RuntimeError(f"rank {path} design hash differs from packer receipt")
        designs.add(info["source_design_sha256"])
        rows.append(info)
    if len(designs) != 1:
        raise RuntimeError(f"layer {layer} ranks carry different design hashes: {sorted(designs)}")
    return {
        "layer": layer,
        "status": "pass",
        "ranks": rows,
        "source_design_sha256": designs.pop(),
        "packer_receipt": {"path": str(Path(packer_receipt).resolve()), "sha256": sha256_file(packer_receipt)},
        "postwrite_receipt": {
            "path": str(Path(postwrite_receipt).resolve()),
            "sha256": sha256_file(postwrite_receipt),
            "schema": postwrite.get("schema"),
        },
        "bytes": sum(row["bytes"] for row in rows),
        "weight_payload_bytes": sum(row["weight_payload_bytes"] for row in rows),
        "metadata_bytes": sum(row["metadata_bytes"] for row in rows),
        "logical_elements": sum(row["logical_elements"] for row in rows),
    }


def cmd_layer_complete(args: argparse.Namespace) -> dict:
    return verify_layer(
        layer=args.layer,
        sidecars=args.sidecars,
        packer_receipt=args.packer_receipt,
        postwrite_receipt=args.postwrite_receipt,
        allowed_design_sha256=set(args.allow_design_sha256),
        strict_geometry=not args.no_strict_geometry,
        rehash=not args.trust_receipt_hashes,
    )


# ------------------------------------------------------------------------ reuse


def cmd_reuse_layer(args: argparse.Namespace) -> dict:
    layer = args.layer
    packer = load_packer_receipt(args.source_packer_receipt, layer)
    load_postwrite_receipt(args.source_postwrite_receipt, layer)
    target_packer = Path(args.receipts) / f"layer-{layer:03d}-sidecars.json"
    target_postwrite = Path(args.receipts) / f"layer-{layer:03d}-postwrite.json"
    target_reuse = Path(args.receipts) / f"layer-{layer:03d}-reuse.json"
    for target in (target_packer, target_postwrite, target_reuse):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
    Path(args.sidecars).mkdir(parents=True, exist_ok=True)
    Path(args.receipts).mkdir(parents=True, exist_ok=True)
    links = []
    for entry in packer["ranks"]:
        rank = int(entry["rank"])
        source = Path(args.source_dir) / rank_path(Path("."), layer, rank).name
        if not source.is_file():
            raise RuntimeError(f"missing source sidecar {source}")
        digest = sha256_file(source)
        if digest != entry["sha256"] or source.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"source sidecar {source} differs from its packer receipt")
        target = rank_path(args.sidecars, layer, rank)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        mode = "hardlink"
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
            mode = "copy"
        if sha256_file(target) != digest:
            raise RuntimeError(f"reused sidecar {target} does not hash like its source")
        links.append({"rank": rank, "source": str(source.resolve()), "target": str(target.resolve()), "mode": mode, "sha256": digest, "bytes": int(entry["bytes"])})
    shutil.copyfile(args.source_packer_receipt, target_packer)
    shutil.copyfile(args.source_postwrite_receipt, target_postwrite)
    record = {
        "schema": "glm53-p8-full-coupled-layer-reuse.v1",
        "layer": layer,
        "reused_at": utc_now(),
        "reason": args.reason,
        "source_dir": str(Path(args.source_dir).resolve()),
        "source_packer_receipt": {"path": str(Path(args.source_packer_receipt).resolve()), "sha256": sha256_file(args.source_packer_receipt)},
        "source_postwrite_receipt": {"path": str(Path(args.source_postwrite_receipt).resolve()), "sha256": sha256_file(args.source_postwrite_receipt)},
        "links": links,
    }
    target_reuse.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


# --------------------------------------------------------------------- manifest


def cmd_manifest(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    allowed = set(args.allow_design_sha256)
    layers = []
    totals = {"bytes": 0, "weight_payload_bytes": 0, "metadata_bytes": 0, "logical_elements": 0}
    for layer in LAYERS:
        receipts = Path(args.receipts)
        row = verify_layer(
            layer=layer,
            sidecars=args.sidecars,
            packer_receipt=receipts / f"layer-{layer:03d}-sidecars.json",
            postwrite_receipt=receipts / f"layer-{layer:03d}-postwrite.json",
            allowed_design_sha256=allowed,
            strict_geometry=not args.no_strict_geometry,
            rehash=not args.trust_receipt_hashes,
        )
        reuse = receipts / f"layer-{layer:03d}-reuse.json"
        if reuse.is_file():
            row["reuse_receipt"] = {"path": str(reuse.resolve()), "sha256": sha256_file(reuse)}
        for key in totals:
            totals[key] += row[key]
        layers.append(row)
    payload_bpw = totals["weight_payload_bytes"] * 8 / totals["logical_elements"]
    if totals["weight_payload_bytes"] * 32 != 17 * totals["logical_elements"]:
        raise RuntimeError(f"aggregate weight payload is {payload_bpw} bpw, expected exactly 4.25")
    designs = {}
    for row in layers:
        designs.setdefault(row["source_design_sha256"], []).append(row["layer"])
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": utc_now(),
        "layer_range": [LAYERS[0], LAYERS[-1]],
        "world_size": WORLD_SIZE,
        "codec": {
            "alphabet": "e4m3",
            "bits": 4,
            "law": "procedural-mcg-alpha2",
            "ldlq": False,
            "scale": "ue8m0-k32",
            "boundary": COUPLED_BOUNDARY,
        },
        "designs": {digest: sorted(rows) for digest, rows in designs.items()},
        "transform": {"path": str(Path(args.transform).resolve()), "sha256": sha256_file(args.transform)},
        "layers": layers,
        "payload": {
            "weight_payload_bytes": totals["weight_payload_bytes"],
            "weight_payload_bpw": payload_bpw,
            "coupled_metadata_bytes": totals["metadata_bytes"],
            "coupled_metadata_bpw": totals["metadata_bytes"] * 8 / totals["logical_elements"],
            "safetensors_header_bytes": totals["bytes"] - totals["weight_payload_bytes"] - totals["metadata_bytes"],
            "file_bytes": totals["bytes"],
            "stored_bpw_including_metadata": (totals["weight_payload_bytes"] + totals["metadata_bytes"]) * 8 / totals["logical_elements"],
            "logical_elements": totals["logical_elements"],
        },
        "identity_reference": {
            "manifest": str(Path(args.identity_manifest).resolve()) if args.identity_manifest else None,
            "sha256": sha256_file(args.identity_manifest) if args.identity_manifest else None,
        },
        "protected_boundary": "fit capture only; no selection, confirmation, final, or reserved confirmation logits opened",
        "isa_cost": "P8 E4M3 mxf8f6f4 has twice the NVFP4 MMA issue count; no speed claim",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"manifest": str(output.resolve()), "sha256": sha256_file(output), "payload": manifest["payload"]}


# ----------------------------------------------------------------------- ledger


def cmd_ledger(args: argparse.Namespace) -> dict:
    row = {
        "at": utc_now(),
        "unix": time.time(),
        "note": args.note,
        "paths": paths_apparent_bytes(args.campaign_path),
        "hard_links_charged_once": True,
        "filesystem_free_bytes": shutil.disk_usage(args.root).free,
        "max_new_bytes": args.max_new_bytes,
    }
    row["campaign_apparent_bytes"] = sum(row["paths"].values())
    with Path(args.output).open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return row


# -------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    guard = sub.add_parser("guard")
    guard.add_argument("--root", type=Path, required=True, help="directory on the target filesystem")
    guard.add_argument("--campaign-path", type=Path, action="append", required=True)
    guard.add_argument("--max-new-bytes", type=int, required=True)
    guard.add_argument("--need-bytes", type=int, required=True)
    guard.add_argument("--min-free-bytes", type=int, default=10_000_000_000)
    guard.add_argument("--gpus", default="0,1,2,3")
    guard.set_defaults(func=cmd_guard)

    reuse = sub.add_parser("reuse-layer")
    reuse.add_argument("--layer", type=int, required=True, choices=LAYERS)
    reuse.add_argument("--source-dir", type=Path, required=True)
    reuse.add_argument("--source-packer-receipt", type=Path, required=True)
    reuse.add_argument("--source-postwrite-receipt", type=Path, required=True)
    reuse.add_argument("--sidecars", type=Path, required=True)
    reuse.add_argument("--receipts", type=Path, required=True)
    reuse.add_argument("--reason", default="same transform, scales, encoder and rate as the full build")
    reuse.set_defaults(func=cmd_reuse_layer)

    complete = sub.add_parser("layer-complete")
    complete.add_argument("--layer", type=int, required=True, choices=LAYERS)
    complete.add_argument("--sidecars", type=Path, required=True)
    complete.add_argument("--packer-receipt", type=Path, required=True)
    complete.add_argument("--postwrite-receipt", type=Path, required=True)
    complete.add_argument("--allow-design-sha256", action="append", required=True)
    complete.add_argument("--no-strict-geometry", action="store_true")
    complete.add_argument("--trust-receipt-hashes", action="store_true")
    complete.set_defaults(func=cmd_layer_complete)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--sidecars", type=Path, required=True)
    manifest.add_argument("--receipts", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--transform", type=Path, required=True)
    manifest.add_argument("--identity-manifest", type=Path)
    manifest.add_argument("--allow-design-sha256", action="append", required=True)
    manifest.add_argument("--no-strict-geometry", action="store_true")
    manifest.add_argument("--trust-receipt-hashes", action="store_true")
    manifest.set_defaults(func=cmd_manifest)

    ledger = sub.add_parser("ledger")
    ledger.add_argument("--root", type=Path, required=True)
    ledger.add_argument("--campaign-path", type=Path, action="append", required=True)
    ledger.add_argument("--max-new-bytes", type=int, required=True)
    ledger.add_argument("--output", type=Path, required=True)
    ledger.add_argument("--note", default="")
    ledger.set_defaults(func=cmd_ledger)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.func(args)
    except (RuntimeError, FileExistsError, FileNotFoundError, ValueError) as error:
        print(json.dumps({"status": "fail", "command": args.command, "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
