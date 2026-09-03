"""Memory-map sealed BF16 activation/router captures and expose preregistered fit samples."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class RoutedSample:
    rows: np.ndarray
    weights: np.ndarray


class LayerCapture:
    def __init__(self, capture_root: Path, layer: int, roles_path: Path, max_samples: int = 256):
        self.root = capture_root
        self.layer = layer
        self.max_samples = max_samples
        self.manifest = json.loads((capture_root / "capture-manifest.json").read_text())
        roles = json.loads(roles_path.read_text())["roles"]
        fit_items = {item["id"]: item for item in roles["fit"]}
        self.fit_ids = set(fit_items)
        windows = {item["window_id"]: item for item in self.manifest["windows"]}
        missing = self.fit_ids - windows.keys()
        if missing:
            raise ValueError(f"fit windows absent from capture: {sorted(missing)}")
        mismatched = sorted(
            window_id
            for window_id, role_item in fit_items.items()
            if role_item["input_sha256"] != windows[window_id]["token_ids_sha256"]
        )
        if mismatched:
            raise ValueError(f"fit window token identities disagree with capture manifest: {mismatched}")
        self.window_indices = sorted(windows[wid]["window_index"] for wid in self.fit_ids)
        abi = self.manifest["file_abi"]
        self.rows, self.hidden_size = abi["hidden_bf16"]["shape"]
        _, self.top_k = abi["topk_ids_u16le"]["shape"]
        info = self.manifest["files"][str(layer)]
        layer_root = capture_root / f"layers/layer-{layer:03d}"
        self.hidden_path = layer_root / "hidden.bf16.bin"
        self.ids_path = layer_root / "topk_ids.u16le.bin"
        self.weights_path = layer_root / "topk_weights.f32le.bin"
        for path, key in ((self.hidden_path, "hidden_bf16"), (self.ids_path, "topk_ids_u16le"), (self.weights_path, "topk_weights_f32le")):
            expected = info[key]["bytes"]
            if not path.is_file() or path.stat().st_size != expected:
                raise FileNotFoundError(f"missing or incomplete capture: {path} expected_bytes={expected}")
        self.hidden_words = np.memmap(self.hidden_path, mode="r", dtype="<u2", shape=(self.rows, self.hidden_size))
        self.topk_ids = np.memmap(self.ids_path, mode="r", dtype="<u2", shape=(self.rows, self.top_k))
        self.topk_weights = np.memmap(self.weights_path, mode="r", dtype="<f4", shape=(self.rows, self.top_k))
        self._routes = self._index_routes()

    def _index_routes(self) -> dict[int, RoutedSample]:
        row_parts: dict[int, list[np.ndarray]] = {expert: [] for expert in range(288)}
        weight_parts: dict[int, list[np.ndarray]] = {expert: [] for expert in range(288)}
        for window in self.window_indices:
            start = window * 2048
            ids = np.asarray(self.topk_ids[start : start + 2048])
            weights = np.asarray(self.topk_weights[start : start + 2048])
            for expert in np.unique(ids):
                token, slot = np.nonzero(ids == expert)
                row_parts[int(expert)].append(start + token.astype(np.int64))
                weight_parts[int(expert)].append(weights[token, slot].astype(np.float32))
        result = {}
        for expert in range(288):
            rows = np.concatenate(row_parts[expert]) if row_parts[expert] else np.empty(0, dtype=np.int64)
            weights = np.concatenate(weight_parts[expert]) if weight_parts[expert] else np.empty(0, dtype=np.float32)
            # Stable capture order is part of the ABI; take the first routed samples without score inspection.
            result[expert] = RoutedSample(rows[: self.max_samples], weights[: self.max_samples])
        return result

    def samples(self, expert: int) -> tuple[torch.Tensor, torch.Tensor]:
        route = self._routes[expert]
        if len(route.rows) < 16:
            raise ValueError(f"expert {expert} has only {len(route.rows)} selected routes")
        words = np.array(self.hidden_words[route.rows], copy=True)
        hidden = torch.from_numpy(words).view(torch.bfloat16)
        return hidden, torch.from_numpy(np.array(route.weights, copy=True))

    def counts(self) -> dict[int, int]:
        return {expert: len(route.rows) for expert, route in self._routes.items()}
