"""Memory-map sealed BF16 activation/router captures and expose preregistered fit samples."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def _balanced_route_sample(
    rows_by_domain: dict[str, np.ndarray],
    weights_by_domain: dict[str, np.ndarray],
    start: int,
    count: int,
) -> RoutedSample:
    """Select stable round-robin routes so early capture order cannot dominate fit."""
    domains = sorted(rows_by_domain)
    positions = {domain: 0 for domain in domains}
    selected_rows: list[int] = []
    selected_weights: list[float] = []
    stop = start + count
    while len(selected_rows) < stop:
        advanced = False
        for domain in domains:
            position = positions[domain]
            domain_rows = rows_by_domain[domain]
            if position >= len(domain_rows):
                continue
            selected_rows.append(int(domain_rows[position]))
            selected_weights.append(float(weights_by_domain[domain][position]))
            positions[domain] = position + 1
            advanced = True
            if len(selected_rows) == stop:
                break
        if not advanced:
            break
    return RoutedSample(
        np.asarray(selected_rows[start:], dtype=np.int64),
        np.asarray(selected_weights[start:], dtype=np.float32),
    )


@dataclass(frozen=True)
class RoutedSample:
    rows: np.ndarray
    weights: np.ndarray


class LayerCapture:
    def __init__(
        self,
        capture_root: Path,
        layer: int,
        roles_path: Path,
        max_samples: int = 256,
        data_role: str = "fit",
        sample_offset: int = 0,
        sampling_strategy: str = "stable",
    ):
        self.root = capture_root
        self.layer = layer
        self.max_samples = max_samples
        if sample_offset < 0:
            raise ValueError("sample offset must be nonnegative")
        self.sample_offset = sample_offset
        if sampling_strategy not in ("stable", "domain-balanced"):
            raise ValueError(f"unknown sampling strategy: {sampling_strategy}")
        self.sampling_strategy = sampling_strategy
        self.manifest = json.loads((capture_root / "capture-manifest.json").read_text())
        roles = json.loads(roles_path.read_text())["roles"]
        if data_role not in roles:
            raise ValueError(f"unknown capture data role: {data_role}")
        role_items = {item["id"]: item for item in roles[data_role]}
        self.data_role = data_role
        self.fit_ids = set(role_items)
        windows = {item["window_id"]: item for item in self.manifest["windows"]}
        missing = self.fit_ids - windows.keys()
        if missing:
            raise ValueError(f"fit windows absent from capture: {sorted(missing)}")
        mismatched = sorted(
            window_id
            for window_id, role_item in role_items.items()
            if role_item["input_sha256"] != windows[window_id]["token_ids_sha256"]
        )
        if mismatched:
            raise ValueError(f"fit window token identities disagree with capture manifest: {mismatched}")
        self.window_indices = sorted(windows[wid]["window_index"] for wid in self.fit_ids)
        self.window_domains = {
            windows[window_id]["window_index"]: role_items[window_id].get("domain", "unknown")
            for window_id in self.fit_ids
        }
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
        self._validate_materialized_windows(len(windows))
        self._routes = self._index_routes()

    def _validate_materialized_windows(self, window_count: int) -> None:
        """Fail closed when a role points at an unmaterialized capture hole."""
        if window_count <= 0 or self.rows % window_count:
            raise ValueError(
                f"capture rows cannot be partitioned into windows: rows={self.rows} "
                f"windows={window_count}"
            )
        rows_per_window = self.rows // window_count
        for window in self.window_indices:
            start = window * rows_per_window
            stop = start + rows_per_window
            # BF16 absolute-value ordering is identical to the unsigned bit
            # pattern after clearing the sign bit. This avoids materializing a
            # second multi-gigabyte floating-point capture merely to detect an
            # all-zero sparse hole.
            magnitude_words = np.bitwise_and(
                np.asarray(self.hidden_words[start:stop]), np.uint16(0x7FFF)
            )
            nonzero_count = int(np.count_nonzero(magnitude_words))
            absmax_bits = int(magnitude_words.max(initial=0))
            if nonzero_count == 0 or absmax_bits == 0:
                raise ValueError(
                    "selected capture window is not materialized: "
                    f"layer={self.layer} window_index={window} "
                    f"nonzero_count={nonzero_count} absmax_bf16_bits={absmax_bits}"
                )

    def _index_routes(self) -> dict[int, RoutedSample]:
        row_parts: dict[int, list[np.ndarray]] = {expert: [] for expert in range(288)}
        weight_parts: dict[int, list[np.ndarray]] = {expert: [] for expert in range(288)}
        domain_row_parts: dict[int, dict[str, list[np.ndarray]]] = {
            expert: {} for expert in range(288)
        }
        domain_weight_parts: dict[int, dict[str, list[np.ndarray]]] = {
            expert: {} for expert in range(288)
        }
        for window in self.window_indices:
            domain = self.window_domains[window]
            start = window * 2048
            ids = np.asarray(self.topk_ids[start : start + 2048])
            weights = np.asarray(self.topk_weights[start : start + 2048])
            for expert in np.unique(ids):
                token, slot = np.nonzero(ids == expert)
                row_parts[int(expert)].append(start + token.astype(np.int64))
                weight_parts[int(expert)].append(weights[token, slot].astype(np.float32))
                domain_row_parts[int(expert)].setdefault(domain, []).append(
                    start + token.astype(np.int64)
                )
                domain_weight_parts[int(expert)].setdefault(domain, []).append(
                    weights[token, slot].astype(np.float32)
                )
        result = {}
        for expert in range(288):
            rows = np.concatenate(row_parts[expert]) if row_parts[expert] else np.empty(0, dtype=np.int64)
            weights = np.concatenate(weight_parts[expert]) if weight_parts[expert] else np.empty(0, dtype=np.float32)
            start = self.sample_offset
            stop = start + self.max_samples
            if self.sampling_strategy == "stable":
                # Stable capture order is part of the original ABI.
                result[expert] = RoutedSample(rows[start:stop], weights[start:stop])
            else:
                rows_by_domain = {
                    domain: np.concatenate(parts)
                    for domain, parts in domain_row_parts[expert].items()
                }
                weights_by_domain = {
                    domain: np.concatenate(parts)
                    for domain, parts in domain_weight_parts[expert].items()
                }
                result[expert] = _balanced_route_sample(
                    rows_by_domain,
                    weights_by_domain,
                    start,
                    self.max_samples,
                )
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
