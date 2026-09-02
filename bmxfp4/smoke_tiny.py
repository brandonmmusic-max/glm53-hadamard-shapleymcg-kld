"""End-to-end smoke test on a tiny random Qwen3-MoE model (catches hook/order/shape bugs before the real run).

Builds a 2-layer, 8-expert model (hidden 256, moe_intermediate 64, vocab 512), saves it as an HF checkpoint,
writes a tiny seal with random windows of 64 tokens, then runs: teacher capture -> hessians -> arm (RTN + GPTQ,
identity + had16 + random, A16 + A4) -> candidates -> allocation, using the real pipeline scripts with env overrides.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/1d791492-949d-44a1-ba3e-f2d339bf8385/scratchpad/bmxfp4-tiny")
CODE = Path(__file__).parent
PY = "/home/brandonmusic/klc-env/bin/python"


def build():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(vocab_size=512, hidden_size=256, intermediate_size=512, moe_intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=64, num_experts=8, num_experts_per_tok=2,
                         norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[], max_position_embeddings=256, tie_word_embeddings=False)
    torch.manual_seed(0)
    m = Qwen3MoeForCausalLM(cfg).to(torch.bfloat16)
    mdir = ROOT / "models" / "tiny"; mdir.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(mdir, safe_serialization=True)
    rng = np.random.default_rng(0)
    def wins(n): return [{"document_id": f"d{i}", "domain": "axis1_general", "offset": 0, "token_ids": rng.integers(0, 512, size=64).tolist(), "token_sha256": ""} for i in range(n)]
    seal = {"role_counts": {"fit": 6, "selection": 3, "confirmation": 3, "final": 3}, "window_tokens": 64,
            "windows": {"fit": wins(6), "selection": wins(3), "confirmation": wins(3), "final": wins(3)}}
    (ROOT / "seal.json").write_text(json.dumps(seal))
    return mdir


def run(cmd, env):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
    print("\n".join("   " + l for l in tail), flush=True)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {cmd[1]}")


def main():
    mdir = build()
    env = dict(os.environ, BMXFP4_ROOT=str(ROOT), BMXFP4_MODEL=str(mdir), BMXFP4_SEAL_JSON=str(ROOT / "seal.json"), CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "3"))
    run([PY, str(CODE / "run_teacher.py"), "--device", "cuda:0", "--panels", "selection,final,confirmation,wikitext", "--b0-runs", "2"], env)
    run([PY, str(CODE / "run_hessians.py"), "--device", "cuda:0", "--layers", "0-1"], env)
    run([PY, str(CODE / "run_arm.py"), "--arm", "tiny-B3", "--device", "cuda:0", "--method", "rtn", "--panels", "selection", "--a4"], env)
    run([PY, str(CODE / "run_arm.py"), "--arm", "tiny-B3p", "--device", "cuda:0", "--method", "gptq", "--panels", "selection,final", "--a4", "--save-weights"], env)
    run([PY, str(CODE / "run_arm.py"), "--arm", "tiny-B5", "--device", "cuda:0", "--method", "gptq", "--rotation", "had16", "--permutation", "diag_band", "--panels", "selection", "--a4"], env)
    run([PY, str(CODE / "run_arm.py"), "--arm", "tiny-B4s1", "--device", "cuda:0", "--method", "gptq", "--rotation", "random:1", "--panels", "selection"], env)
    run([PY, str(CODE / "run_arm.py"), "--arm", "tiny-cand", "--device", "cuda:0", "--mode", "candidates", "--method", "gptq"], env)
    run([PY, str(CODE / "run_arm.py"), "--arm", "tiny-calib0", "--device", "cuda:0", "--mode", "calib", "--calib-layer", "0"], env)
    run([PY, str(CODE / "allocate.py"), "--candidates", str(ROOT / "arms" / "tiny-cand" / "candidates"), "--out", str(ROOT / "arms" / "alloc-tiny"), "--budget-bpw", "5.0"], env)
    run([PY, str(CODE / "run_arm.py"), "--arm", "tiny-B7", "--device", "cuda:0", "--method", "gptq", "--tiermap", str(ROOT / "arms" / "alloc-tiny" / "tiermap.json"), "--panels", "selection"], env)
    print("TINY SMOKE OK")


if __name__ == "__main__":
    main()
