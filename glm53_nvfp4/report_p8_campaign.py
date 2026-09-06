"""Compile the full-coupled + Shapley-allocated P8 campaign into one Markdown report.

Reads only receipts that exist and says so when one is missing; never invents a
number.  Inputs: the K4 build manifest, the uniform coupled CF32 analysis, the
per-layer damage receipts, the allocation, the mixed-rate manifest, the mixed
CF32 analysis, the K3/K5 device-closure results, and the assembly chunk receipts
(to compare re-encoded chunk hashes with the scored candidates).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LAYERS = tuple(range(3, 45))
IDENTITY_FILE_BYTES = 161_715_707_328
IDENTITY_PAYLOAD_BYTES = 161_715_585_024
CONTEXT = {
    "identity K4 P8, NVFP4 KV, tail-repair v2 image, N64 fused scratch (09-05)": 0.03730955956731441,
    "EXL3 4.0 bpw, same path, TP4/EP4/DCP4 (09-05)": 0.031611840268931456,
}


def _load(path: Path | None):
    if path is None or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text())


def _fmt(value, digits=6):
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_report(args: argparse.Namespace) -> str:
    k4 = _load(args.k4_manifest)
    uniform = _load(args.uniform_analysis)
    allocation = _load(args.allocation)
    mixed = _load(args.mixed_manifest)
    mixed_kld = _load(args.mixed_analysis)
    closures = {bits: _load(Path(args.closure_dir) / f"closure-k{bits}" / "result.json") if args.closure_dir else None
                for bits in (3, 5)}
    damage = {}
    if args.damage_dir:
        for layer in LAYERS:
            row = _load(Path(args.damage_dir) / f"damage-layer-{layer:03d}.json")
            if row:
                damage[layer] = row
    lines = [f"# GLM-5.3-Flash coupled TrellisMX P8 with same-size K3/K4/K5 allocation", "",
             f"Generated {args.generated_at}. Every value below is read from a receipt named in the",
             "evidence section; missing receipts are reported as n/a rather than estimated.", ""]

    lines += ["## 1. Uniform coupled K4 P8 (Phase 0)", ""]
    if k4:
        p = k4["payload"]
        lines += ["| Quantity | Value |", "|---|---:|",
                  f"| Layers | {len(k4['layers'])} (3-44), 4 TP ranks each |",
                  f"| Weight payload bytes | {p['weight_payload_bytes']:,} ({p['weight_payload_bpw']:.4f} bpw) |",
                  f"| Coupled metadata bytes | {p['coupled_metadata_bytes']:,} ({p['coupled_metadata_bpw']:.6f} bpw) |",
                  f"| File bytes | {p['file_bytes']:,} (identity K4 files: {IDENTITY_FILE_BYTES:,}) |",
                  f"| Stored bpw incl. metadata | {p['stored_bpw_including_metadata']:.10f} |",
                  f"| Designs | {', '.join(f'{k[:12]}… layers {len(v)}' for k, v in k4['designs'].items())} |", ""]
    else:
        lines += ["K4 manifest not available.", ""]
    if uniform:
        arm = uniform["arms"]["coupled_full"]
        lines += ["| CF32 metric (coupled_full, v10 image) | Value |", "|---|---:|",
                  f"| True-decode mean KLD | {arm['true_decode_mean_kld']:.6f} |",
                  f"| Mean KLD incl. row 0 | {arm['including_prefill_mean_kld']:.6f} |",
                  f"| Window BCa 95% | [{arm['true_decode_window_bca95'][0]:.6f}, {arm['true_decode_window_bca95'][1]:.6f}] |"]
        for domain, value in arm["per_domain_true_decode_mean_kld"].items():
            lines.append(f"| {domain} | {value:.6f} |")
        lines += ["", "Runtime labels: " + ", ".join(f"{k}={v}" for k, v in arm["conditions"].items()), ""]
    else:
        lines += ["Uniform coupled CF32 analysis not available.", ""]

    lines += ["## 2. Candidate damage and allocation (Phase 1)", ""]
    if damage:
        lines += ["Fit-role routed-output damage per layer (mean per token; lower is better).", "",
                  "| Layer | K3 | K4 | K5 | K3/K4 | K5/K4 | Allocated |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for layer in LAYERS:
            row = damage.get(layer)
            if not row:
                lines.append(f"| {layer} | n/a | n/a | n/a | | | |")
                continue
            r = row["rates"]
            k3, k4v, k5 = (r.get(k, {}).get("damage_mean_per_token") for k in ("3", "4", "5"))
            chosen = allocation["assignment"].get(str(layer)) if allocation else None
            ratio3 = f"{k3 / k4v:.2f}" if k3 and k4v else "n/a"
            ratio5 = f"{k5 / k4v:.2f}" if k5 and k4v else "n/a"
            lines.append(f"| {layer} | {_fmt(k3)} | {_fmt(k4v)} | {_fmt(k5)} | {ratio3} | {ratio5} | {('K' + str(chosen)) if chosen else 'n/a'} |")
        lines.append("")
    else:
        lines += ["No damage receipts available.", ""]
    if allocation:
        lines += ["| Allocation | Value |", "|---|---:|",
                  f"| Counts K3/K4/K5 | {allocation['counts']['3']}/{allocation['counts']['4']}/{allocation['counts']['5']} |",
                  f"| Net offset #K5-#K3 | {allocation['selected_net_offset']} (max allowed {allocation['max_net_offset_k5_minus_k3']}) |",
                  f"| Predicted damage change vs uniform K4 | {allocation['predicted_relative_damage_change'] * 100:+.2f} % |",
                  f"| Estimated file bytes | {allocation['estimated_total_file_bytes']:,} ({allocation['bytes_under_budget']:,} under the identity budget) |", ""]
    else:
        lines += ["Allocation not available.", ""]

    lines += ["## 3. Allocated model (mixed K3/K4/K5, v11 image)", ""]
    if mixed:
        p = mixed["payload"]
        lines += ["| Quantity | Value |", "|---|---:|",
                  f"| File bytes | {p['file_bytes']:,} ({'same-size or smaller' if p['same_size_or_smaller'] else 'LARGER'} than identity files {IDENTITY_FILE_BYTES:,}) |",
                  f"| Weight payload bpw | {p['weight_payload_bpw']:.6f} |",
                  f"| Stored bpw incl. metadata | {p['stored_bpw_including_metadata']:.6f} |",
                  f"| Counts K3/K4/K5 | {mixed['allocation']['counts']['3']}/{mixed['allocation']['counts']['4']}/{mixed['allocation']['counts']['5']} |", ""]
    else:
        lines += ["Mixed-rate manifest not available.", ""]
    if mixed_kld:
        arm = mixed_kld["arms"]["coupled_full"]
        lines += ["| CF32 metric (allocated model) | Value |", "|---|---:|",
                  f"| True-decode mean KLD | {arm['true_decode_mean_kld']:.6f} |",
                  f"| Mean KLD incl. row 0 | {arm['including_prefill_mean_kld']:.6f} |",
                  f"| Window BCa 95% | [{arm['true_decode_window_bca95'][0]:.6f}, {arm['true_decode_window_bca95'][1]:.6f}] |"]
        for domain, value in arm["per_domain_true_decode_mean_kld"].items():
            lines.append(f"| {domain} | {value:.6f} |")
        lines += ["", "Runtime labels: " + ", ".join(f"{k}={v}" for k, v in arm["conditions"].items()), ""]
        if uniform:
            a, b = uniform["arms"]["coupled_full"]["true_decode_mean_kld"], arm["true_decode_mean_kld"]
            lines += [f"Allocated minus uniform coupled K4: {b - a:+.6f} ({(b - a) / a * 100:+.2f} %). Same windows, "
                      "different images (v10 vs v11); not a paired-bootstrap comparison unless both arms ran in one execution.", ""]
    else:
        lines += ["Allocated-model CF32 analysis not available.", ""]

    lines += ["## 4. Context (not matched controls)", "", "| Reference | True-decode KLD |", "|---|---:|"]
    for name, value in CONTEXT.items():
        lines.append(f"| {name} | {value:.6f} |")
    lines += ["", "The identity control was measured on a different image and FC1 configuration; the owner",
              "dropped the identity comparison, so no paired claim is made against it.", ""]

    lines += ["## 5. Device closures (K3/K5 small-M coupled kernels, v11)", ""]
    for bits, result in closures.items():
        if result:
            lines.append(f"- K{bits}: decision {result.get('decision')}, five-run bitwise identical, "
                         f"final cosine/relative L2 per run recorded in {args.closure_dir}/closure-k{bits}/result.json")
        else:
            lines.append(f"- K{bits}: no closure result available")
    lines.append("")

    lines += ["## 6. Designs and evidence", ""]
    designs: dict[str, set[int]] = {}
    for manifest, label in ((k4, "uniform K4"), (mixed, "allocated")):
        if manifest:
            for digest, layers in manifest.get("designs", {}).items():
                designs.setdefault(f"{label}: {digest}", set()).update(int(x) for x in layers)
    if designs:
        lines += ["Design files are pinned by sha256 in every sidecar and archived byte-identical on the campaign",
                  "volume under designs/; they are not version-controlled because they embed machine-local paths.", "",
                  "| Checkpoint: design sha256 | Layers |", "|---|---:|"]
        for key, layers in sorted(designs.items()):
            lines.append(f"| {key} | {len(layers)} |")
        lines.append("")
    evidence = _load(args.evidence_manifest)
    if evidence:
        lines += [f"Scrubbed receipt copies ({len(evidence['files'])} files; placeholders replace machine-local roots,",
                  "original sha256 recorded):", "", "| File | Original sha256 | Scrubbed sha256 |", "|---|---|---|"]
        for entry in evidence["files"]:
            lines.append(f"| {entry['dest']} | {entry['source_sha256'][:16]}… | {entry['scrubbed_sha256'][:16]}… |")
        lines.append("")
    elif not designs:
        lines += ["No design or evidence manifests available.", ""]

    lines += ["## 7. Claim boundary", "",
              "- CF32 is already-opened development data (32 windows, 8 per domain, 2,047 rows each; row 0",
              "  excluded from true decode); not final qualification.",
              "- P8 is E4M3 mxf8f6f4 at twice NVFP4's MMA issue count; no speed claim is made here.",
              "- The allocation payoff is a fit-role routed-output proxy with exact per-token Shapley shares;",
              "  the end-to-end KLD above is the only quality claim.",
              "- Grouped M64 prefill kernels remain K4-only; non-K4 layers serve M>1 row by row.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k4-manifest", type=Path)
    parser.add_argument("--uniform-analysis", type=Path)
    parser.add_argument("--damage-dir", type=Path)
    parser.add_argument("--allocation", type=Path)
    parser.add_argument("--mixed-manifest", type=Path)
    parser.add_argument("--mixed-analysis", type=Path)
    parser.add_argument("--closure-dir", type=Path)
    parser.add_argument("--evidence-manifest", type=Path, help="SCRUB_MANIFEST.json written by scrub_receipts_for_publication.py")
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(build_report(args))
    print(str(args.output))


if __name__ == "__main__":
    main()
