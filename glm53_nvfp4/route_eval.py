"""Capture logical routed-expert choices for an already opened role.

This diagnostic deliberately does not read teacher logits.  It uses vLLM's
server-side ``enable_return_routed_experts`` transport and records only the
logical top-k ids for routed layers 3..44.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from .shard_index import sha256_file


LAYER_START = 3
LAYER_STOP = 45
TOP_K = 8
NUM_EXPERTS = 288


def decode_routed_experts(encoded: str, expected_tokens: int) -> np.ndarray:
    raw = base64.b64decode(encoded, validate=True)
    routes = np.load(io.BytesIO(raw), allow_pickle=False)
    if routes.ndim != 3:
        raise RuntimeError(f"routed experts must be rank 3, got {routes.shape}")
    if routes.shape[0] != expected_tokens or routes.shape[1] < LAYER_STOP:
        raise RuntimeError(
            f"unexpected routed-expert geometry {routes.shape}; expected "
            f"({expected_tokens}, >= {LAYER_STOP}, {TOP_K})"
        )
    if routes.shape[2] != TOP_K:
        raise RuntimeError(f"unexpected top-k width: {routes.shape[2]}")
    routed = np.asarray(routes[:, LAYER_START:LAYER_STOP, :], dtype=np.uint16)
    if int(routed.max(initial=0)) >= NUM_EXPERTS:
        raise RuntimeError("captured routed-expert id exceeds GLM expert count")
    if np.any(np.diff(np.sort(routed, axis=-1), axis=-1) == 0):
        raise RuntimeError("captured top-k contains a duplicate logical expert")
    return np.ascontiguousarray(routed)


def _post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("conditional-fit", "selection"), required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    roles = json.loads(args.roles.read_text())
    windows = roles["roles"][args.role]
    if not windows:
        raise RuntimeError(f"role {args.role!r} is empty")
    output = args.output
    route_dir = output.parent / f"{output.stem}-routes"
    fresh = {
        "schema": "glm53-route-capture.v1",
        "run_id": args.run_id,
        "role": args.role,
        "config_id": args.config_id,
        "roles_sha256": sha256_file(args.roles),
        "model_name": args.model_name,
        "layer_range": [LAYER_START, LAYER_STOP],
        "top_k": TOP_K,
        "expert_count": NUM_EXPERTS,
        "windows": {},
        "started_unix": time.time(),
    }
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        manifest = json.loads(output.read_text())
        for key in ("run_id", "role", "config_id", "roles_sha256", "model_name"):
            if manifest.get(key) != fresh[key]:
                raise RuntimeError(f"resume identity mismatch: {key}")
    else:
        manifest = fresh
    route_dir.mkdir(parents=True, exist_ok=True)

    for window in windows:
        if args.resume and window["id"] in manifest["windows"]:
            continue
        token_path = Path(window["token_path"])
        if sha256_file(token_path) != window["input_sha256"]:
            raise RuntimeError(f"token identity mismatch: {window['id']}")
        tokens = np.load(token_path, allow_pickle=False)
        if tokens.shape != (window["prediction_positions"] + 1,):
            raise RuntimeError(f"token geometry mismatch: {window['id']}")
        started = time.monotonic()
        response = _post(
            args.url,
            {
                "model": args.model_name,
                "prompt": [int(value) for value in tokens],
                "max_tokens": 1,
                "temperature": 0.0,
                "routed_experts_prompt_start": 0,
            },
            args.timeout,
        )
        choices = response.get("choices", [])
        if len(choices) != 1 or not choices[0].get("routed_experts"):
            raise RuntimeError(f"server returned no routed experts: {window['id']}")
        # vLLM returns routing for every prompt token.  KLD has one fewer
        # next-token prediction because the final token has no target, but the
        # causal route diagnostic must preserve the server's exact prompt-row
        # geometry and compares like-for-like between arms.
        routes = decode_routed_experts(
            choices[0]["routed_experts"], expected_tokens=tokens.size
        )
        route_path = route_dir / f"{window['id']}.npy"
        if route_path.exists():
            raise FileExistsError(f"refusing to overwrite {route_path}")
        np.save(route_path, routes, allow_pickle=False)
        manifest["windows"][window["id"]] = {
            "domain": window["domain"],
            "input_sha256": window["input_sha256"],
            "route_path": str(route_path),
            "route_bytes": route_path.stat().st_size,
            "route_sha256": sha256_file(route_path),
            "shape": list(routes.shape),
            "dtype": str(routes.dtype),
            "elapsed_seconds": time.monotonic() - started,
            "response_id": response.get("id"),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"window": window["id"], "route_sha256": manifest["windows"][window["id"]]["route_sha256"]}), flush=True)

    if set(manifest["windows"]) != {window["id"] for window in windows}:
        raise RuntimeError("route capture did not complete the exact role")
    manifest["status"] = "complete"
    manifest["finished_unix"] = time.time()
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run_id": args.run_id, "windows": len(windows)}, sort_keys=True))


if __name__ == "__main__":
    main()
