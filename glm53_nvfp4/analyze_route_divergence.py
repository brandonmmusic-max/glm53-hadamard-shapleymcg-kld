"""Analyze causal downstream routing divergence from the layer-22 candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .shard_index import sha256_file


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rank_average(x), _rank_average(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _load_routes(manifest: dict, window_id: str) -> np.ndarray:
    item = manifest["windows"][window_id]
    path = Path(item["route_path"])
    if sha256_file(path) != item["route_sha256"]:
        raise RuntimeError(f"route hash mismatch: {path}")
    routes = np.load(path, allow_pickle=False)
    if list(routes.shape) != item["shape"] or str(routes.dtype) != item["dtype"]:
        raise RuntimeError(f"route geometry mismatch: {path}")
    return routes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--candidate-routes", type=Path, required=True)
    parser.add_argument("--baseline-routes", type=Path, required=True)
    parser.add_argument("--candidate-kld", type=Path, required=True)
    parser.add_argument("--baseline-kld", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    roles = json.loads(args.roles.read_text())
    candidate_manifest = json.loads(args.candidate_routes.read_text())
    baseline_manifest = json.loads(args.baseline_routes.read_text())
    candidate_kld = json.loads(args.candidate_kld.read_text())
    baseline_kld = json.loads(args.baseline_kld.read_text())
    expected = [item["id"] for item in roles["roles"]["conditional-fit"]]
    for manifest in (candidate_manifest, baseline_manifest):
        if manifest.get("status") != "complete" or set(manifest["windows"]) != set(expected):
            raise RuntimeError("route manifest does not cover exact role")
    for run in (candidate_kld, baseline_kld):
        if run.get("status") != "complete" or set(run["windows"]) != set(expected):
            raise RuntimeError("KLD manifest does not cover exact role")

    layers = list(range(3, 45))
    set_changed_parts = []
    slot_changed_parts = []
    per_window = []
    first_divergence_counts = {str(layer): 0 for layer in layers}
    first_divergence_counts["none"] = 0
    for window_id in expected:
        candidate = _load_routes(candidate_manifest, window_id)
        baseline = _load_routes(baseline_manifest, window_id)
        if candidate.shape != baseline.shape or candidate.shape[1:] != (42, 8):
            raise RuntimeError(f"route shape mismatch for {window_id}")
        sorted_candidate = np.sort(candidate, axis=-1)
        sorted_baseline = np.sort(baseline, axis=-1)
        set_changed = np.any(sorted_candidate != sorted_baseline, axis=-1)
        slot_changed = candidate != baseline
        set_changed_parts.append(set_changed)
        slot_changed_parts.append(slot_changed)
        any_by_token = set_changed.any(axis=1)
        first_index = np.argmax(set_changed, axis=1)
        for changed, index in zip(any_by_token, first_index, strict=True):
            key = str(layers[int(index)]) if changed else "none"
            first_divergence_counts[key] += 1
        post_rate = float(set_changed[:, 20:].mean())
        delta_kld = float(
            candidate_kld["windows"][window_id]["mean_kld"]
            - baseline_kld["windows"][window_id]["mean_kld"]
        )
        per_window.append(
            {
                "window_id": window_id,
                "domain": candidate_manifest["windows"][window_id]["domain"],
                "post_layer22_set_change_rate": post_rate,
                "paired_delta_kld": delta_kld,
            }
        )

    set_changed_all = np.concatenate(set_changed_parts, axis=0)
    slot_changed_all = np.concatenate(slot_changed_parts, axis=0)
    pre_change_rate = float(set_changed_all[:, :20].mean())
    post_change_rate = float(set_changed_all[:, 20:].mean())
    per_layer = [
        {
            "layer": layer,
            "top8_set_change_rate": float(set_changed_all[:, index].mean()),
            "ordered_slot_change_rate": float(slot_changed_all[:, index].mean()),
        }
        for index, layer in enumerate(layers)
    ]
    correlation = spearman(
        np.asarray([row["post_layer22_set_change_rate"] for row in per_window]),
        np.asarray([row["paired_delta_kld"] for row in per_window]),
    )
    valid = pre_change_rate == 0.0
    causal = valid and post_change_rate > 0.0 and all(
        count == 0 for layer, count in first_divergence_counts.items()
        if layer != "none" and int(layer) <= 22
    )
    material = causal and post_change_rate >= 0.005 and correlation >= 0.30
    decision = (
        "invalid-pre-layer-divergence"
        if not valid
        else "support-router-sensitive-objective"
        if material
        else "routing-diverges-but-does-not-meet-materiality"
        if causal
        else "no-causal-routing-divergence"
    )
    payload = {
        "schema": "glm53-layer22-route-divergence-analysis.v1",
        "plan_sha256": sha256_file(args.plan),
        "roles_sha256": sha256_file(args.roles),
        "candidate_route_manifest_sha256": sha256_file(args.candidate_routes),
        "baseline_route_manifest_sha256": sha256_file(args.baseline_routes),
        "windows": len(expected),
        "tokens": int(set_changed_all.shape[0]),
        "pre_through_layer22_set_change_rate": pre_change_rate,
        "post_layer22_set_change_rate": post_change_rate,
        "spearman_post_route_divergence_vs_delta_kld": correlation,
        "first_divergence_counts": first_divergence_counts,
        "per_layer": per_layer,
        "per_window": per_window,
        "decision": decision,
        "confirmation_logits_opened": False,
        "teacher_logits_read": False,
        "ldlq_used": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "windows", "tokens", "pre_through_layer22_set_change_rate",
        "post_layer22_set_change_rate", "spearman_post_route_divergence_vs_delta_kld",
        "decision",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
