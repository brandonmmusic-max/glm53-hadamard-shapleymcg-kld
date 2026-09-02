"""Round-2 smoke test on the tiny model built by smoke_tiny.py: fit teacher -> attribution (path-integrated)
-> calibrated allocation -> causal re-encode with re-anchors -> calibrated rotation learning."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/tmp/bmxfp4-tiny")
CODE = Path(__file__).parent
PY = "/home/brandonmusic/klc-env/bin/python"


def run(cmd, env):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-8:]
    print("\n".join("   " + l for l in tail), flush=True)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {cmd[1]}")


def main():
    mdir = ROOT / "models" / "tiny"
    env = dict(os.environ, BMXFP4_ROOT=str(ROOT), BMXFP4_MODEL=str(mdir), BMXFP4_SEAL_JSON=str(ROOT / "seal.json"), CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "3"))
    arms = ROOT / "arms"
    run([PY, str(CODE / "run_teacher.py"), "--device", "cuda:0", "--panels", "fit", "--b0-runs", "1"], env)
    run([PY, str(CODE / "run_arm.py"), "--arm", "tiny-prov", "--device", "cuda:0", "--method", "gptq", "--rotation", "had16", "--panels", "selection", "--save-weights"], env)
    run([PY, str(CODE / "attribution.py"), "--endpoint", str(arms / "tiny-prov"), "--device", "cuda:0", "--windows", "0-2", "--chunk", "1", "--out", str(arms / "attrib-tiny" / "part-0.json")], env)
    run([PY, str(CODE / "attribution.py"), "--endpoint", str(arms / "tiny-prov"), "--device", "cuda:0", "--windows", "3-5", "--chunk", "1", "--out", str(arms / "attrib-tiny" / "part-1.json")], env)
    run([PY, str(CODE / "attribution.py"), "--merge", str(arms / "attrib-tiny"), "--out", str(arms / "attrib-tiny" / "attribution.json")], env)
    run([PY, str(CODE / "run_arm.py"), "--arm", "tiny-cand48", "--device", "cuda:0", "--mode", "candidates", "--method", "gptq", "--rotation", "had16"], env)
    run([PY, str(CODE / "allocate.py"), "--candidates", str(arms / "tiny-cand48" / "candidates"), "--out", str(arms / "alloc-tiny-att"), "--budget-bpw", "5.0",
         "--attribution", str(arms / "attrib-tiny" / "attribution.json")], env)
    run([PY, str(CODE / "run_causal.py"), "--arm", "tiny-causal", "--device", "cuda:0", "--rotation", "had16", "--tiermap", str(arms / "alloc-tiny-att" / "tiermap.json"),
         "--panels", "selection,final", "--a4", "--reanchor", "1", "--save-weights"], env)
    run([PY, str(CODE / "learn_rotation_v2.py"), "--device", "cuda:0", "--layers", "0-1", "--out", str(arms / "learnedR-cal"), "--steps", "5", "--batch", "4"], env)
    run([PY, str(CODE / "run_arm.py"), "--arm", "tiny-B5L", "--device", "cuda:0", "--method", "gptq", "--rotation", f"learned:{arms / 'learnedR-cal'}", "--panels", "selection"], env)
    print("ROUND-2 SMOKE OK")


if __name__ == "__main__":
    main()
