"""Build zero-copy matched BF16 overlays for a frozen layer-coalition game."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path

from .shard_index import sha256_file


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.")


def _parse_layer_overlay(value: str) -> tuple[int, Path]:
    raw_layer, separator, raw_path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("expected LAYER=/absolute/overlay")
    path = Path(raw_path).resolve()
    if not (path / "model.safetensors.index.json").is_file():
        raise argparse.ArgumentTypeError(f"missing overlay index: {path}")
    return int(raw_layer), path


def _layer_entries(weight_map: dict[str, str], layer: int) -> dict[str, str]:
    result = {}
    for name, shard in weight_map.items():
        match = LAYER_RE.search(name)
        if match and int(match.group(1)) == layer and ".mlp.experts." in name:
            result[name] = shard
    if not result:
        raise RuntimeError(f"overlay contains no routed-expert entries for layer {layer}")
    return result


def _sources(path: Path) -> dict[str, Path]:
    result = {}
    for item in path.iterdir():
        if item.name in {"model.safetensors.index.json", "OVERLAY.json"}:
            continue
        result[item.name] = item.resolve()
    overlay_path = path / "OVERLAY.json"
    if overlay_path.is_file():
        overlay = json.loads(overlay_path.read_text())
        for raw in overlay.get("chunks", []):
            item = Path(raw).resolve()
            result[item.name] = item
    return result


def _merge_sources(paths: list[Path]) -> dict[str, Path]:
    merged: dict[str, Path] = {}
    for path in paths:
        for name, source in _sources(path).items():
            previous = merged.get(name)
            if previous is not None and previous != source:
                if not name.endswith(".safetensors"):
                    # Configuration/tokenizer files come from the carrier;
                    # overlay-local copies are irrelevant to the weight map.
                    continue
                raise RuntimeError(f"ambiguous shard basename {name}: {previous} vs {source}")
            merged[name] = source
    return merged


def _replace_layer(weight_map: dict[str, str], layer: int, entries: dict[str, str]) -> None:
    old = set(_layer_entries(weight_map, layer))
    new = set(entries)
    # A BF16 overlay removes ModelOpt scale/input-scale keys.  Replacing the
    # complete routed-expert layer entry set preserves that fail-closed ABI.
    for name in old - new:
        del weight_map[name]
    weight_map.update(entries)


def _remove_quantized_layers(payload: dict, layers: list[int]) -> dict:
    """Return a ModelOpt config with routed experts for ``layers`` unquantized."""
    result = copy.deepcopy(payload)
    quant = result.get("quantization_config", result)
    targets = {
        f"model.language_model.layers.{layer}.mlp.experts" for layer in layers
    }
    for group in quant.get("config_groups", {}).values():
        if "targets" in group:
            group["targets"] = [
                target for target in group["targets"] if target not in targets
            ]
    quantized_layers = quant.get("quantized_layers", {})
    for target in targets:
        quantized_layers.pop(target, None)
    return result


def _write_overlay(
    output: Path,
    *,
    carrier: Path,
    index_template: dict,
    weight_map: dict[str, str],
    sources: dict[str, Path],
    base_layers: list[int],
    candidate_layers: list[int],
    design_sha256: str,
    generated_configs: dict[str, dict],
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    required_shards = set(weight_map.values())
    missing = sorted(required_shards - set(sources))
    if missing:
        raise RuntimeError(f"missing shard sources: {missing[:8]}")
    broken = sorted(
        name
        for name in required_shards
        if not sources[name].is_file()
    )
    if broken:
        raise RuntimeError(f"missing shard source files: {broken[:8]}")
    for name, source in sources.items():
        if name in generated_configs:
            continue
        if name in required_shards or source.parent == carrier:
            os.symlink(source, output / name)
    config_hashes = {}
    for name, payload in generated_configs.items():
        config_path = output / name
        config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        config_hashes[name] = sha256_file(config_path)
    index = dict(index_template)
    index["weight_map"] = weight_map
    index_path = output / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    overlay = {
        "schema": "glm53-p8.kld-shapley-zero-copy-overlay.v1",
        "carrier": str(carrier),
        "base_layers": base_layers,
        "candidate_layers": candidate_layers,
        "required_load_format": "instanttensor",
        "design_sha256": design_sha256,
        "index_sha256": sha256_file(index_path),
        "generated_config_sha256": config_hashes,
        "zero_copy": True,
        "ldlq": False,
    }
    overlay_path = output / "OVERLAY.json"
    overlay_path.write_text(json.dumps(overlay, indent=2, sort_keys=True) + "\n")
    return {**overlay, "overlay_sha256": sha256_file(overlay_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--base-overlay", type=_parse_layer_overlay, action="append", required=True)
    parser.add_argument("--candidate-overlay", type=_parse_layer_overlay, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite overlay matrix or manifest")
    design = json.loads(args.design.read_text())
    if design.get("ldlq") is not False:
        raise RuntimeError("design does not freeze the no-LDLQ boundary")
    layers = list(map(int, design["layers"]))
    base_paths = dict(args.base_overlay)
    candidate_paths = dict(args.candidate_overlay)
    if set(base_paths) != set(layers) or set(candidate_paths) != set(layers):
        raise RuntimeError("base and candidate overlays must cover every design layer exactly once")

    carrier = args.carrier.resolve()
    original = json.loads((carrier / "model.safetensors.index.json").read_text())
    base_map = dict(original["weight_map"])
    base_entries = {}
    candidate_entries = {}
    for layer in layers:
        base_index = json.loads((base_paths[layer] / "model.safetensors.index.json").read_text())
        candidate_index = json.loads((candidate_paths[layer] / "model.safetensors.index.json").read_text())
        base_entries[layer] = _layer_entries(base_index["weight_map"], layer)
        candidate_entries[layer] = _layer_entries(candidate_index["weight_map"], layer)
        _replace_layer(base_map, layer, base_entries[layer])

    all_paths = [carrier, *base_paths.values(), *candidate_paths.values()]
    sources = _merge_sources(all_paths)
    design_hash = sha256_file(args.design)
    generated_configs = {}
    for name in ("config.json", "hf_quant_config.json"):
        path = carrier / name
        if path.is_file():
            generated_configs[name] = _remove_quantized_layers(
                json.loads(path.read_text()), layers
            )
    rows = []
    args.output_root.mkdir(parents=True)
    for cid, coalition in design["coalitions"].items():
        weight_map = dict(base_map)
        for layer in coalition:
            _replace_layer(weight_map, int(layer), candidate_entries[int(layer)])
        output = args.output_root / cid
        receipt = _write_overlay(
            output,
            carrier=carrier,
            index_template=original,
            weight_map=weight_map,
            sources=sources,
            base_layers=layers,
            candidate_layers=list(map(int, coalition)),
            design_sha256=design_hash,
            generated_configs=generated_configs,
        )
        rows.append(
            {
                "coalition_id": cid,
                "layers": list(map(int, coalition)),
                "model": str(output.resolve()),
                "index_sha256": receipt["index_sha256"],
                "overlay_sha256": receipt["overlay_sha256"],
                "generated_config_sha256": receipt["generated_config_sha256"],
            }
        )
    manifest = {
        "schema": "glm53-p8.kld-shapley-overlay-matrix.v1",
        "design": str(args.design.resolve()),
        "design_sha256": design_hash,
        "carrier": str(carrier),
        "base_overlays": {str(layer): str(path) for layer, path in sorted(base_paths.items())},
        "candidate_overlays": {str(layer): str(path) for layer, path in sorted(candidate_paths.items())},
        "coalitions": rows,
        "zero_copy": True,
        "ldlq": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(args.manifest), "sha256": sha256_file(args.manifest), "coalitions": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
