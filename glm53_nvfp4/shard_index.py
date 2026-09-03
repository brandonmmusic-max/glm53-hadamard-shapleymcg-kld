"""Read indexed safetensors without loading unrelated GLM layers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from safetensors import safe_open


class IndexedCheckpoint:
    def __init__(self, root: Path, index_path: Path | None = None):
        self.root = root.resolve()
        index_path = index_path or (self.root / "model.safetensors.index.json")
        payload = json.loads(index_path.read_text())
        self.weight_map: dict[str, str] = payload["weight_map"]
        self.metadata = payload.get("metadata", {})

    def names(self, prefix: str = "") -> list[str]:
        return sorted(name for name in self.weight_map if name.startswith(prefix))

    def get(self, name: str):
        shard = self.root / self.weight_map[name]
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def layer_names(self, layer: int) -> list[str]:
        needle = f".layers.{layer}."
        return sorted(name for name in self.weight_map if needle in name)

    def expert_prefix(self, layer: int, expert: int) -> str:
        suffix = f"layers.{layer}.mlp.experts.{expert}."
        matches = [name[: -len(name.split(".")[-1])] for name in self.weight_map if suffix in name]
        if not matches:
            raise KeyError(f"no expert tensors for layer={layer} expert={expert}")
        # Strip projection and tensor leaf from an arbitrary matching name.
        name = next(name for name in self.weight_map if suffix in name)
        return name[: name.index(suffix)] + suffix

    def inventory(self, prefix: str = "") -> dict:
        files = sorted({self.weight_map[name] for name in self.names(prefix)})
        return {
            "root": str(self.root),
            "tensors": len(self.names(prefix)),
            "files": [
                {"path": filename, "bytes": (self.root / filename).stat().st_size, "sha256": sha256_file(self.root / filename)}
                for filename in files
            ],
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
