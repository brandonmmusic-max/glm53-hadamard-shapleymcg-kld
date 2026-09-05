"""GPU-free validation of the installed P8 runtime image and amended identities."""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path


def verify_constructor(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoEDynamicKernelBackend")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    args = {n.arg for n in init.args.args + init.args.kwonlyargs}
    required = {"trellis_codebook", "trellis_scaled", "trellis_identity_boundary", "deterministic_output"}
    if not required <= args:
        raise ValueError(f"P8 constructor missing {sorted(required - args)}")


def preflight(amendment_path):
    amendment_path = Path(amendment_path)
    amendment = json.loads(amendment_path.read_text())
    for name, expected in amendment["prior_plans"].items():
        if hashlib.sha256((amendment_path.parent / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"prior plan mismatch: {name}")
    image = amendment["image_id"]
    actual = subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip()
    if actual != image:
        raise ValueError("image identity mismatch")
    for path, expected in amendment["source_sha256"].items():
        source = subprocess.check_output([
            "docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/cat", image, path,
        ])
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError(f"image source mismatch: {path}")
        if path.endswith("/dynamic.py"):
            verify_constructor(source.decode())
    return {"status": "pass", "image_id": image, "gpu_used": False,
            "amendment_sha256": hashlib.sha256(amendment_path.read_bytes()).hexdigest(),
            "source_sha256": amendment["source_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment", required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(args.amendment), sort_keys=True))
