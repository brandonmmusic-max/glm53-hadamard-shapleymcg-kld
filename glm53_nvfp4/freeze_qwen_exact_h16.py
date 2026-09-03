"""Freeze the exact all-projection Qwen H16 candidate before selection."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .shard_index import sha256_file


HARNESS = Path("/home/brandonmusic/KLC_SANDBOXES/glm53-flash-kld-eval")
SUFFIXES = ("weight", "weight_scale", "weight_scale_2", "input_scale")


def _file(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _git(root: Path) -> dict:
    return {
        "root": str(root.resolve()),
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "clean": not bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        ),
    }


def _expected(weight_map: dict[str, str]) -> set[str]:
    result = set()
    for layer in range(3, 45):
        for expert in range(288):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                stem = (
                    f"model.language_model.layers.{layer}.mlp.experts."
                    f"{expert}.{projection}"
                )
                for suffix in SUFFIXES:
                    name = f"{stem}.{suffix}"
                    if name not in weight_map:
                        raise KeyError(name)
                    result.add(name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--full-build", type=Path, required=True)
    parser.add_argument("--conditional-analysis", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--selection-wave", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    implementation, harness = _git(root), _git(HARNESS)
    if not implementation["clean"] or not harness["clean"]:
        raise RuntimeError("implementation and KLD harness must be clean")

    roles = json.loads(args.roles.read_text())
    wave = str(args.selection_wave)
    if wave not in roles["selection_waves"]:
        raise RuntimeError(f"selection wave {wave} is not sealed")
    conditional = json.loads(args.conditional_analysis.read_text())
    if conditional.get("role") != "conditional-fit" or conditional.get("decision") != "pass":
        raise RuntimeError("full candidate did not pass the conditional-fit rule")
    build = json.loads(args.full_build.read_text())
    if build.get("status") != "pass" or build.get("layers") != list(range(3, 45)):
        raise RuntimeError("full-build receipt is incomplete")
    if Path(build["candidate"]).resolve() != args.candidate.resolve():
        raise RuntimeError("full-build candidate identity mismatch")
    if (
        build.get("rotation") != "had16"
        or build.get("rotation_scope") != "all"
        or build.get("calibration_max_samples_per_expert") != 256
    ):
        raise RuntimeError("full-build is not the exact requested Qwen H16 recipe")

    stock_map = json.loads(
        (args.stock / "model.safetensors.index.json").read_text()
    )["weight_map"]
    candidate_map = json.loads(
        (args.candidate / "model.safetensors.index.json").read_text()
    )["weight_map"]
    expected = _expected(stock_map)
    changed = {name for name in stock_map if candidate_map[name] != stock_map[name]}
    if changed != expected:
        raise RuntimeError(f"candidate changed {len(changed)} rather than {len(expected)} tensors")
    overlay = json.loads((args.candidate / "OVERLAY.json").read_text())
    if (
        overlay.get("required_load_format") != "instanttensor"
        or overlay.get("redirected_tensors") != len(expected)
    ):
        raise RuntimeError("candidate overlay does not implement indexed replacement")
    chunks = {candidate_map[name] for name in expected}
    if len(chunks) != 42 * 5:
        raise RuntimeError(f"expected 210 layer chunks, found {len(chunks)}")
    for name in chunks:
        if not (args.candidate / name).is_file():
            raise RuntimeError(f"missing candidate chunk {name}")

    payload = {
        "schema": "glm53-nvfp4-v10.qwen-exact-h16-selection-freeze.v1",
        "status": "pass",
        "selection_wave_authorized": args.selection_wave,
        "recipe": {
            "rotation": "normalized Sylvester H16",
            "group_size": 16,
            "rotation_scope": "all",
            "gate_up_shared_R_in": True,
            "independent_R_mid_for_down": True,
            "full_hessian": True,
            "static_act_order_within_group": True,
            "sequential_gptq_column_slab": 128,
            "routed_samples_per_expert_cap": 256,
            "runtime_activation_transform": "inside Humming before each matching GEMM",
            "required_load_format": "instanttensor",
        },
        "candidate": {
            "root": str(args.candidate.resolve()),
            "index": _file(args.candidate / "model.safetensors.index.json"),
            "overlay": _file(args.candidate / "OVERLAY.json"),
            "changed_tensors": len(changed),
            "chunk_files": len(chunks),
        },
        "stock": {
            "root": str(args.stock.resolve()),
            "index": _file(args.stock / "model.safetensors.index.json"),
        },
        "full_build": _file(args.full_build),
        "conditional_fit_analysis": _file(args.conditional_analysis),
        "roles": _file(args.roles),
        "selection_window_ids": roles["selection_waves"][wave],
        "implementation": implementation,
        "analysis_harness": harness,
        "analysis_files": [
            _file(root / "glm53_nvfp4/role_eval.py"),
            _file(root / "glm53_nvfp4/paired_role_analysis.py"),
            _file(root / "runtime_patch/sitecustomize.py"),
            _file(root / "scripts/run_kld_v3.sh"),
        ],
        "quantization_files": [
            _file(root / "glm53_nvfp4/quantize_layer.py"),
            _file(root / "glm53_nvfp4/block_gptq.py"),
            _file(root / "glm53_nvfp4/block_rotation.py"),
            _file(root / "glm53_nvfp4/calibrate_input_scale.py"),
            _file(root / "glm53_nvfp4/candidate.py"),
            _file(root / "glm53_nvfp4/validate_layer.py"),
            _file(root / "scripts/run_qwen_exact_had16_full_v9.sh"),
            _file(root / "scripts/run_qwen_exact_h16_qualification_v10.sh"),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "pass",
                "selection_wave": args.selection_wave,
                "changed_tensors": len(changed),
                "implementation_commit": implementation["commit"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
