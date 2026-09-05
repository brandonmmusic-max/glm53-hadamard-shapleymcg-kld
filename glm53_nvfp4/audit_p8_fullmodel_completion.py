"""Check full-layer dispatch and token-level completeness without changing KLD."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .shard_index import sha256_file


def verify_dispatch(text: str, design_sha: str) -> dict:
    expected = {(layer, rank) for layer in range(3, 45) for rank in range(4)}
    events = {}
    for event in ("WEIGHTS_READY", "FORWARD"):
        rows = re.findall(r"GLM53_P8_NATIVE_" + event + r" layer=(\d+) rank=(\d+) ([^\r\n]*)", text)
        pairs = {(int(layer), int(rank)) for layer, rank, _ in rows}
        if pairs != expected:
            raise ValueError(f"{event} dispatch inventory mismatch: missing={sorted(expected-pairs)}, extra={sorted(pairs-expected)}")
        for layer, rank, fields in rows:
            tokens = dict(re.findall(r"(\w+)=([^\s]+)", fields))
            want = {"stream": "K4", "alphabet": "E4M3", "scale": "UE8M0_K32", "boundary": "identity", "ldlq": "false"}
            if event == "WEIGHTS_READY":
                want.update(design_sha256=design_sha, law="mcg")
            else:
                want.update(mma="mxf8f6f4", law="procedural_mcg", deterministic="route_topk_sum", physical_bpw="4.25")
            if any(tokens.get(key) != value for key, value in want.items()):
                raise ValueError(f"{event} wrong native contract at layer={layer}, rank={rank}")
        events[event.lower()] = len(pairs)
    return events


def verify_records(rows: list[dict], records: Path, manifest: dict, roles_sha: str) -> dict:
    expected = {row["id"]: row for row in rows}
    if len(expected) != 32 or len(rows) != 32 or sorted(Counter(row["domain"] for row in rows).values()) != [8]*4:
        raise ValueError("expected 32 distinct domain-balanced windows")
    if manifest.get("status") != "complete" or manifest.get("role") != "conditional-fit" or manifest.get("roles_sha256") != roles_sha:
        raise ValueError("run manifest is incomplete or role identity differs")
    if set(manifest.get("windows", {})) != set(expected):
        raise ValueError("run manifest window inventory mismatch")
    verified = {}
    for path in sorted(records.glob("*.parquet")):
        table = pq.read_table(path)
        if table.num_rows == 0:
            raise ValueError(f"empty record: {path.name}")
        ids = set(table["window_id"].to_pylist())
        if len(ids) != 1:
            raise ValueError("mixed window identities")
        key = ids.pop()
        if key not in expected or key in verified:
            raise ValueError("unexpected or duplicate window")
        row = expected[key]
        entry = manifest["windows"][key]
        digest = sha256_file(path)
        if entry["record_sha256"] != digest or Path(entry["record_path"]).resolve() != path.resolve():
            raise ValueError(f"record differs from run receipt: {key}")
        for column, value in (("run_id", manifest["run_id"]), ("config_id", manifest["config_id"]), ("domain", row["domain"]), ("context_len", row["prediction_positions"]+1)):
            if set(table[column].to_pylist()) != {value}:
                raise ValueError(f"{key}: {column} identity mismatch")
        metadata = table.schema.metadata or {}
        for field, value in (("role", "conditional-fit"), ("roles_sha256", roles_sha), ("teacher_sha256", row["teacher_sha256"]), ("student_logits_sha256", entry["student_logits_sha256"])):
            if metadata.get(field.encode()) != value.encode():
                raise ValueError(f"{key}: {field} metadata mismatch")
        pos = table["pos"].to_numpy()
        if not np.array_equal(pos, np.arange(row["prediction_positions"])):
            raise ValueError(f"{key}: missing, duplicate, or unordered causal positions")
        kld = table["kld"].to_numpy()
        if not np.isfinite(kld).all() or (kld < 0).any():
            raise ValueError(f"{key}: non-finite or negative token KLD")
        mean = float(kld.mean())
        if not math.isclose(mean, entry["mean_kld"], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"{key}: token mean differs from manifest")
        verified[key] = {"sha256": digest, "positions": len(pos), "mean_kld": mean}
    if set(verified) != set(expected):
        raise ValueError("record window inventory mismatch")
    return verified


def audit(plan_path: Path, roles_path: Path, manifest_path: Path, analysis_path: Path, log_path: Path, records: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    roles_sha = sha256_file(roles_path)
    if roles_sha != plan["role"]["sha256"]:
        raise ValueError("role hash differs from plan")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("run_id") != plan["run_id"]:
        raise ValueError("run identity differs from plan")
    dispatch = verify_dispatch(log_path.read_text(errors="replace"), plan["runtime"]["design_sha256"])
    values = verify_records(json.loads(roles_path.read_text())["roles"]["conditional-fit"], records, manifest, roles_sha)
    analysis = json.loads(analysis_path.read_text())
    if analysis.get("status") != "measurement-complete" or analysis.get("windows") != 32 or analysis.get("plan_sha256") != sha256_file(plan_path) or analysis.get("roles_sha256") != roles_sha:
        raise ValueError("analysis completion or identity mismatch")
    by_id = {row["window_id"]: row for row in analysis["window_records"]}
    if len(analysis["window_records"]) != 32 or set(by_id) != set(values):
        raise ValueError("analysis window inventory mismatch")
    for key, value in values.items():
        if not math.isclose(by_id[key]["candidate_mean_kld"], value["mean_kld"], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"analysis disagrees with token data: {key}")
    mean = float(np.mean([value["mean_kld"] for value in values.values()]))
    if not math.isclose(analysis["candidate_mean_kld"], mean, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("analysis aggregate mismatch")
    return {"schema": "glm53-p8-fullmodel-completion-audit.v1", "status": "pass", "run_id": manifest["run_id"], "windows": 32, "causal_positions": sum(v["positions"] for v in values.values()), "native_dispatch": dispatch, "candidate_mean_kld": mean, "records": values, "inputs": {str(p): sha256_file(p) for p in (plan_path, roles_path, manifest_path, analysis_path, log_path)}, "limitation": "Dispatch receipts and complete KLD records establish execution coverage; they do not establish a quality or speed win."}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("plan", "roles", "manifest", "analysis", "log", "records", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.plan, args.roles, args.manifest, args.analysis, args.log, args.records)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n"
    if args.output.exists():
        if args.output.read_text() != encoded:
            raise ValueError("refusing to replace a different audit receipt")
    else:
        with args.output.open("x") as handle:
            handle.write(encoded)
    print(json.dumps({k: result[k] for k in ("status", "windows", "causal_positions", "native_dispatch", "candidate_mean_kld")}))


if __name__ == "__main__":
    main()
