"""Build a sparse BF16 pseudoquant overlay from identity/H128 expert payloads."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

from safetensors import safe_open

from .shard_index import sha256_file


EXPERT_RE = re.compile(r"\.experts\.(\d+)\.(?:gate_proj|up_proj|down_proj)\.weight$")


def _index_chunks(chunks: list[Path]) -> tuple[dict[str, Path], int]:
    result: dict[str, Path] = {}
    logical = 0
    for chunk in chunks:
        with safe_open(str(chunk), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if EXPERT_RE.search(name) is None or handle.get_slice(name).get_dtype() != "BF16":
                    raise RuntimeError(f"invalid decoded expert tensor: {name}")
                if name in result:
                    raise RuntimeError(f"duplicate decoded expert tensor: {name}")
                result[name] = chunk
                logical += math.prod(handle.get_slice(name).get_shape())
    return result, logical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--h128-chunk", type=Path, action="append", required=True)
    parser.add_argument("--identity-chunk", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    policy = json.loads(args.policy.read_text())
    schema = policy.get("schema")
    if policy.get("phase") != "selection" or policy.get("ldlq_used") is not False:
        raise RuntimeError("invalid no-LDLQ boundary policy")
    if schema == "glm53-p8-identity-h128-boundary-policy.v1":
        if policy.get("policy_bytes") != 39:
            raise RuntimeError("invalid V1 policy byte count")
        states = [int(expert in set(policy["h128_experts"])) for expert in range(288)]
    elif schema == "glm53-p8-joint-routed-sum-fc1-policy.v2":
        if (
            policy.get("policy_bytes") != 75
            or policy.get("state_names")
            != ["identity", "full-h128", "fc1-h128-identity-down"]
        ):
            raise RuntimeError("invalid V2 joint policy contract")
        states = [int(value) for value in policy["expert_states"]]
        if len(states) != 288 or any(value not in (0, 1, 2) for value in states):
            raise RuntimeError("invalid V2 expert states")
    else:
        raise RuntimeError("unsupported no-LDLQ boundary policy")
    h128_experts = {expert for expert, state in enumerate(states) if state != 0}
    full_h128_experts = {expert for expert, state in enumerate(states) if state == 1}
    h128, h128_logical = _index_chunks(args.h128_chunk)
    identity, identity_logical = _index_chunks(args.identity_chunk)
    if set(h128) != set(identity) or h128_logical != identity_logical:
        raise RuntimeError("identity and H128 decoded payloads do not exactly match")

    args.output.mkdir(parents=True)
    original = json.loads((args.carrier / "model.safetensors.index.json").read_text())
    weight_map = dict(original["weight_map"])
    for item in args.carrier.iterdir():
        if item.name in {"model.safetensors.index.json", "config.json", "hf_quant_config.json", "OVERLAY.json"} or item.name.startswith(".cache"):
            continue
        os.symlink(item.resolve(), args.output / item.name)
    for chunk in args.h128_chunk + args.identity_chunk:
        target = args.output / chunk.name
        if not target.exists():
            os.symlink(chunk.resolve(), target)

    h128_tensor_count = 0
    for name in sorted(h128):
        match = EXPERT_RE.search(name)
        assert match is not None
        expert = int(match.group(1))
        projection = name.rsplit(".", 2)[1]
        state = states[expert]
        selected = h128 if state == 1 or (state == 2 and projection != "down_proj") else identity
        weight_map[name] = selected[name].name
        h128_tensor_count += int(selected is h128)
    removed = []
    for name in list(weight_map):
        if ".layers.3.mlp.experts." in name and name.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
            removed.append(name)
            del weight_map[name]
    updated = dict(original)
    updated["weight_map"] = weight_map
    index_path = args.output / "model.safetensors.index.json"
    index_path.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")

    config = json.loads((args.carrier / "config.json").read_text())
    qc = config["quantization_config"]
    target = "model.language_model.layers.3.mlp.experts"
    del qc["quantized_layers"][target]
    for group in qc.get("config_groups", {}).values():
        group["targets"] = [value for value in group.get("targets", []) if value != target]
    config_path = args.output / "config.json"
    hf_path = args.output / "hf_quant_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    hf_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n")

    physical_bpw = (
        4.25
        + (len(full_h128_experts) / 288.0) * (16.0 * 2048 / (2 * 2048 * 4096 + 4096 * 2048))
        + 8.0 * policy["policy_bytes"] / h128_logical
    )
    overlay = {
        "schema": (
            "glm53-p8-joint-routed-sum-fc1-policy-overlay.v2"
            if schema == "glm53-p8-joint-routed-sum-fc1-policy.v2"
            else "glm53-p8-identity-h128-policy-overlay.v1"
        ),
        "carrier": str(args.carrier.resolve()),
        "policy": str(args.policy.resolve()),
        "policy_sha256": sha256_file(args.policy),
        "bf16_layer": 3,
        "h128_experts": len(h128_experts),
        "full_h128_experts": len(full_h128_experts),
        "fc1_h128_experts": sum(state == 2 for state in states),
        "identity_experts": 288 - len(h128_experts),
        "redirected_tensors": len(h128),
        "h128_redirected_tensors": h128_tensor_count,
        "removed_quantization_tensors": len(removed),
        "physical_codec_bpw": physical_bpw,
        "dense_carrier_bpw": 16.0,
        "required_load_format": "instanttensor",
        "runtime_interpretation": "decoded BF16 pseudoquant carrier; physical codec receipt is separate",
        "ldlq": False,
    }
    overlay_path = args.output / "OVERLAY.json"
    overlay_path.write_text(json.dumps(overlay, indent=2, sort_keys=True) + "\n")
    receipt = {
        **overlay,
        "h128_chunks": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.h128_chunk],
        "identity_chunks": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.identity_chunk],
        "logical_elements": h128_logical,
        "index_sha256": sha256_file(index_path),
        "config_sha256": sha256_file(config_path),
        "hf_quant_config_sha256": sha256_file(hf_path),
        "overlay_sha256": sha256_file(overlay_path),
    }
    receipt_path = args.output / "P8_POLICY_OVERLAY_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
