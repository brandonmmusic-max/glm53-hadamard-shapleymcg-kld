"""Turn the campaign's measured KLD arms into the rate-swap evidence table.

Four measured arms, all on the same sealed 32 windows and the same teacher:

    uniform     every routed layer at K4
    k3_only     the allocation's K3 layers downgraded, its K5 layers left at K4
    k5_only     the allocation's K5 layers upgraded, its K3 layers left at K4
    allocated   both arms installed together (the same-size checkpoint)

From these it reports:

* the downgrade cost and the upgrade benefit, each as a KLD delta against uniform;
* whether the same-size swap paid, from the allocated arm alone;
* group-level additivity: the combined delta divided by the sum of the two solo deltas,
  the quantity measured at 78.2% on the GLM-5.2 campaign.  A retention far from 1 means
  the two groups interact and only the joint measurement may be quoted;
* calibration of the local routed-output damage proxy: what the proxy predicted for each
  arm against what the KLD actually did, so the proxy's usefulness is stated, not assumed.

Every number is read from an analysis receipt; missing arms are reported as absent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "glm53.rate-swap-evidence.v1"
ARMS = ("uniform", "k3_only", "k5_only", "allocated")


def _load(path: Path | None) -> dict | None:
    if path is None or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text())


def _arm(analysis: dict | None) -> dict | None:
    if not analysis:
        return None
    arm = analysis["arms"]["coupled_full"]
    return {"true_decode_mean_kld": arm["true_decode_mean_kld"],
            "including_prefill_mean_kld": arm.get("including_prefill_mean_kld"),
            "window_bca95": arm.get("true_decode_window_bca95"),
            "per_domain": arm.get("per_domain_true_decode_mean_kld"),
            "conditions": arm.get("conditions")}


def _predicted_damage_delta(damage: dict[int, dict], assignment: dict[str, int], rates: tuple[int, ...]) -> float | None:
    """Sum of measured local damage change for the layers moved to the given rates."""
    total = 0.0
    for layer_str, rate in assignment.items():
        if rate not in rates or rate == 4:
            continue
        row = damage.get(int(layer_str))
        if not row:
            return None
        try:
            total += float(row["rates"][str(rate)]["damage_sum"]) - float(row["rates"]["4"]["damage_sum"])
        except KeyError:
            return None
    return total


def build(args: argparse.Namespace) -> dict:
    analyses = {name: _arm(_load(getattr(args, f"{name}_analysis"))) for name in ARMS}
    allocation = _load(args.allocation)
    damage: dict[int, dict] = {}
    for directory in (args.damage_dir, args.k4_only_dir):
        if not directory:
            continue
        for path in sorted(Path(directory).glob("damage-layer-*.json")):
            row = json.loads(path.read_text())
            damage.setdefault(int(row["layer"]), row)

    result: dict = {"schema": SCHEMA, "arms": analyses, "protocol": "CF32, 32 windows, 2047 rows each, row 0 excluded"}
    base = analyses["uniform"]
    if base:
        deltas = {}
        for name in ("k3_only", "k5_only", "allocated"):
            other = analyses[name]
            deltas[name] = (other["true_decode_mean_kld"] - base["true_decode_mean_kld"]) if other else None
        result["kld_delta_vs_uniform"] = deltas
        cost, gain, both = deltas["k3_only"], deltas["k5_only"], deltas["allocated"]
        if cost is not None and gain is not None:
            solo_sum = cost + gain
            result["group_effects"] = {
                "downgrade_cost_kld": cost,
                "upgrade_benefit_kld": -gain if gain is not None else None,
                "solo_sum_kld": solo_sum,
                "combined_kld": both,
                "additivity_retention": (both / solo_sum) if (both is not None and solo_sum) else None,
                "note": "retention is combined delta over the sum of the two solo deltas; 1.0 is exactly additive, "
                        "the GLM-5.2 two-layer campaign measured 0.782",
            }
        if both is not None:
            result["verdict"] = {
                "same_size_swap_pays": bool(both < 0),
                "kld_change": both,
                "relative_change": both / base["true_decode_mean_kld"],
                "statement": ("the same-size K3/K5 reallocation lowered end-to-end KLD" if both < 0 else
                              "the same-size K3/K5 reallocation did not lower end-to-end KLD; uniform K4 is the "
                              "better checkpoint at this budget"),
            }
    if allocation and damage:
        assignment = allocation["assignment"]
        predicted = {"k3_only": _predicted_damage_delta(damage, assignment, (3,)),
                     "k5_only": _predicted_damage_delta(damage, assignment, (5,)),
                     "allocated": _predicted_damage_delta(damage, assignment, (3, 5))}
        result["local_proxy"] = {
            "predicted_routed_output_damage_delta": predicted,
            "measured_kld_delta": result.get("kld_delta_vs_uniform"),
            "units_differ": "the proxy is squared residual-stream error on fit tokens; only the sign and the "
                            "ranking are comparable to KLD, never the magnitude",
            "sign_agreement": {name: (None if (predicted.get(name) is None or (result.get("kld_delta_vs_uniform") or {}).get(name) is None)
                                      else bool((predicted[name] > 0) == (result["kld_delta_vs_uniform"][name] > 0)))
                               for name in ("k3_only", "k5_only", "allocated")},
        }
    if allocation:
        result["allocation"] = {"counts": allocation.get("counts"), "selected_net_offset": allocation.get("selected_net_offset"),
                                "estimated_total_file_bytes": allocation.get("estimated_total_file_bytes"),
                                "bytes_under_budget": allocation.get("bytes_under_budget"),
                                "k3_layers": sorted(int(k) for k, v in allocation["assignment"].items() if v == 3),
                                "k5_layers": sorted(int(k) for k, v in allocation["assignment"].items() if v == 5)}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ARMS:
        parser.add_argument(f"--{name.replace('_', '-')}-analysis", type=Path)
    parser.add_argument("--allocation", type=Path)
    parser.add_argument("--damage-dir", type=Path)
    parser.add_argument("--k4-only-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(args)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"deltas": result.get("kld_delta_vs_uniform"),
                      "verdict": (result.get("verdict") or {}).get("statement"),
                      "additivity_retention": (result.get("group_effects") or {}).get("additivity_retention")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
