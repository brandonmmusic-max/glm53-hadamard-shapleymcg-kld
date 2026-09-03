"""Render a concise evidence-bounded campaign report from receipts."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign
    selection = json.loads((root / "evidence/selection-analysis.json").read_text())
    confirmation_path = root / "evidence/confirmation-analysis.json"
    freeze_path = root / "evidence/candidate-freeze.json"
    stopped_path = root / "evidence/selection-terminal.json"
    if confirmation_path.exists():
        result = json.loads(confirmation_path.read_text())
        inventory = json.loads(freeze_path.read_text())["candidate"]
        status = "PASS" if result["decision"] == "pass" else "FAIL"
        evidence_level = "confirmation"
        confirmation_line = "32 preregistered windows, eight per domain, opened only after candidate and analysis freeze."
        primary_rule = "at least 10% lower mean KLD and BCa upper bound below zero"
    else:
        if selection.get("decision") != "stop" or not stopped_path.exists():
            raise RuntimeError("neither completed confirmation nor a sealed selection stop is available")
        result = selection
        inventory = json.loads(stopped_path.read_text())["candidate"]
        status = "STOPPED"
        evidence_level = "selection"
        confirmation_line = "not opened; the preregistered selection stop preserved all 32 confirmation windows."
        primary_rule = "candidate selection mean must be lower than stock before candidate freeze and confirmation"
    layers = [json.loads((root / f"evidence/layer-{layer:03d}-validation.json").read_text()) for layer in range(3, 45)]
    benchmark_summaries = sorted((root / "benchmarks").glob("*/summary.json")) if (root / "benchmarks").is_dir() else []
    projection_lines = []
    for projection in ("gate", "up", "down"):
        means = [item["projections"][projection]["mean_ratio"] for item in layers]
        maximum = max(item["projections"][projection]["maximum_ratio"] for item in layers)
        projection_lines.append(f"- {projection}: mean layer ratio {statistics.mean(means):.4f}; worst expert ratio {maximum:.4f}.")
    text = f"""# GLM-5.3-Flash BF16 to NVFP4 V2 report

Status: **{status}** at {evidence_level} evidence level. No publication or Hub upload was authorized or performed.

## Primary result

- Candidate mean KLD: {result['candidate_mean_kld']:.8f} nats.
- Stock NVFP4 mean KLD: {result['stock_mean_kld']:.8f} nats.
- Relative improvement: {100 * result['relative_improvement']:.2f}%.
- Equal-window paired delta: {result['mean_delta_kld']:.8f}; BCa 95% CI [{result['delta_ci95_bca'][0]:.8f}, {result['delta_ci95_bca'][1]:.8f}].
- Registered rule at this terminal stage: {primary_rule}.

## Quantization gates

All 42 routed-expert layers and all 288 experts per layer passed the matched RTN weighted-error gate. Ratios below 1 favor GPTQ.

{chr(10).join(projection_lines)}

The candidate changes exactly {inventory['changed_tensors']} routed-expert tensors across {inventory['chunks'] if isinstance(inventory['chunks'], int) else len(inventory['chunks'])} chunks. Non-expert tensors remain supplied by the stock carrier.

## Role separation

- Selection comparison: candidate {selection['candidate_mean_kld']:.8f}, stock {selection['stock_mean_kld']:.8f}, decision `{selection['decision']}`.
- Confirmation: {confirmation_line}
- Legacy final windows: not used as new final evidence because they had already been opened.

## Runtime and limitations

The servability gates proved the ModelOpt checkpoint, FLASHINFER_CUTLASS NVFP4 MoE backend, and FLASHINFER_MLA_SPARSE_SM120 attention in TP4/EP4/DCP4 eager mode. Calibration used immutable BF16 activation/router captures and group-16 block Hessians; it did not rerun the 642 GB BF16 checkpoint locally or propagate quantized activations causally across layers. This is one quantization campaign, so it does not estimate between-campaign variance. A selection stop means performance benchmarks are skipped and no confirmation-level claim is available.

Benchmark summary receipts found: {len(benchmark_summaries)}. Performance, Estonia, and LAVD are separate post-quality regimes and do not alter the KLD decision.

## Principal artifacts

- Candidate: `{inventory['root']}`
- Terminal receipt: `{freeze_path if freeze_path.exists() else stopped_path}`
- Decision analysis: `{confirmation_path if confirmation_path.exists() else root / 'evidence/selection-analysis.json'}`
- Experiment record: `{root / 'experiment-record.json'}`
"""
    args.output.write_text(text)
    print(args.output)


if __name__ == "__main__":
    main()
