"""Reconstruct a turboderp EXL3 checkpoint to BF16 tensors (HF layout) using the pinned exllamav3 reader.

Runs INSIDE the K3 docker image (which carries /opt/exllamav3-python + the compiled extension), on a GPU.
Mirrors shapleymcg/scripts/measure_qwen_turboderp_hybrid_k4.py: only the checkpoint reader is imported
(flash_attn stubbed), each transformer block is loaded, every EXL3 linear is expanded with
get_weight_tensor() (EXL3 stores [in, out]; HF stores [out, in]), and the tensors are written per layer.

usage (in container): python3 exl3_reconstruct.py --model /data/models/turboderp-exl3-5.0bpw --out /data/exl3-recon/5.0bpw
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import torch
from safetensors.torch import save_file

ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")


def import_reader(root: Path):
    package_root = root / "exllamav3"
    if "flash_attn" not in sys.modules:
        stub = types.ModuleType("flash_attn")

        def unavailable(*a, **k):
            raise RuntimeError("flash_attn unavailable in reader mode")

        stub.flash_attn_func = unavailable
        stub.flash_attn_with_kvcache = unavailable
        stub.flash_attn_varlen_func = unavailable
        sys.modules["flash_attn"] = stub
    if "kbnf" not in sys.modules:
        sys.modules["kbnf"] = types.ModuleType("kbnf")
    pkg = types.ModuleType("exllamav3")
    pkg.__file__ = str(package_root / "__init__.py")
    pkg.__package__ = "exllamav3"
    pkg.__path__ = [str(package_root)]
    sys.modules["exllamav3"] = pkg
    mp = types.ModuleType("exllamav3.model")
    mp.__file__ = str(package_root / "model" / "__init__.py")
    mp.__package__ = "exllamav3.model"
    mp.__path__ = [str(package_root / "model")]
    sys.modules["exllamav3.model"] = mp
    from exllamav3.model.config import Config
    mp.Config = Config
    from exllamav3.model.model import Model
    return Config, Model


def linear_bf16(module) -> torch.Tensor:
    w = module.inner.get_weight_tensor().T.contiguous()   # [out, in]
    return w.to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exllamav3-root", type=Path, default=Path("/opt/exllamav3-python"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--experts", type=int, default=128)
    a = ap.parse_args()
    sys.path.insert(0, str(a.exllamav3_root))
    Config, Model = import_reader(a.exllamav3_root)
    config = Config.from_directory(str(a.model))
    quant = json.loads((a.model / "quantization_config.json").read_text())
    qmodel = Model.from_config(config)
    print(json.dumps({"modules": len(qmodel.modules), "quant": quant}), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    manifest = {"model_dir": str(a.model), "quantization_config": quant, "layers": {}}
    t0 = time.time()
    for layer in range(a.layers):
        qblock = qmodel.modules[layer + 1]
        config.stc.begin_deferred_load()
        qblock.load(dev)
        config.stc.end_deferred_load()
        tensors = {}
        for proj in ATTENTION:
            m = qblock.find_module(f"model.layers.{layer}.self_attn.{proj}")
            if m is not None and hasattr(m, "inner") and hasattr(m.inner, "get_weight_tensor"):
                tensors[f"self_attn.{proj}"] = linear_bf16(m).cpu()
        for e in range(a.experts):
            prefix = f"model.layers.{layer}.mlp.experts.{e}"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                m = qblock.find_module(f"{prefix}.{proj}")
                tensors[f"experts.{e}.{proj}"] = linear_bf16(m).cpu()
        qblock.unload()
        torch.cuda.empty_cache()
        p = a.out / f"layer-{layer:03d}.safetensors"
        save_file(tensors, str(p))
        manifest["layers"][layer] = {"file": p.name, "tensors": len(tensors), "attention": [k for k in tensors if k.startswith("self_attn")]}
        print(json.dumps({"layer": layer, "tensors": len(tensors), "elapsed": round(time.time() - t0)}), flush=True)
    # lm_head
    qhead = qmodel.modules[-1]
    config.stc.begin_deferred_load()
    qhead.load(dev)
    config.stc.end_deferred_load()
    if hasattr(qhead, "inner") and hasattr(qhead.inner, "get_weight_tensor"):
        save_file({"lm_head": linear_bf16(qhead).cpu()}, str(a.out / "lm_head.safetensors"))
        manifest["lm_head"] = "lm_head.safetensors"
    qhead.unload()
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"done": True, "elapsed": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
