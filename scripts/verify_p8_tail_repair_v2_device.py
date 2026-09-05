#!/usr/bin/env python3
"""Device-level per-row closure for the P8 tail-after-valid-history patch."""
from __future__ import annotations

import hashlib
import json

import torch

from vllm.models.glm5next.nvidia.ops.kpool_compress import expand_pools_and_append_tail


ROWS = 2047
TOPK = 2048
OUT_COLS = 2051
POOL_SIZE = 4


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for Triton intervention closure")
    device = torch.device("cuda:0")
    lengths = torch.arange(1, ROWS + 1, device=device, dtype=torch.int32)
    groups = torch.arange(TOPK // POOL_SIZE, device=device, dtype=torch.int32)
    pool_lens = torch.div(lengths, POOL_SIZE, rounding_mode="floor")
    pool_ids = torch.where(groups[None, :] < pool_lens[:, None], groups[None, :], -1)
    candidate = expand_pools_and_append_tail(pool_ids, lengths, POOL_SIZE)

    cols = torch.arange(OUT_COLS, device=device, dtype=torch.int32)[None, :]
    expected = torch.where(cols < lengths[:, None], cols, -1)
    if not torch.equal(candidate, expected):
        bad = torch.nonzero(torch.any(candidate != expected, dim=1)).flatten().cpu().tolist()
        raise RuntimeError(f"candidate row semantics differ: {bad[:16]}")

    tail_counts = lengths % POOL_SIZE
    tail_starts = lengths - tail_counts
    baseline_history = (cols < TOPK) & (cols < tail_starts[:, None])
    baseline = torch.where(baseline_history, cols, -1)
    tail_offsets = cols - TOPK
    baseline_tail = (tail_offsets >= 0) & (tail_offsets < tail_counts[:, None])
    baseline = torch.where(baseline_tail, tail_starts[:, None] + tail_offsets, baseline)
    changed = torch.any(candidate != baseline, dim=1)
    expected_changed = tail_counts != 0
    if not torch.equal(changed, expected_changed):
        raise RuntimeError("device per-row selector differs from L mod 4 contract")

    differing_rows = torch.nonzero(changed).flatten().cpu().tolist()
    result = {
        "schema": "glm53.p8-tail-repair-device-row-closure.v2",
        "status": "pass",
        "device": torch.cuda.get_device_name(device),
        "sequence_lengths": [1, ROWS],
        "rows": ROWS,
        "topk_consumed_columns": TOPK,
        "output_columns": OUT_COLS,
        "pool_size": POOL_SIZE,
        "differing_rows_zero_based": differing_rows,
        "differing_rows": len(differing_rows),
        "unchanged_rows": ROWS - len(differing_rows),
        "diff_count_by_L_mod_4": {
            str(mod): int(torch.sum(changed & (tail_counts == mod)).item()) for mod in range(4)
        },
        "candidate_sha256": tensor_sha(candidate),
        "baseline_sha256": tensor_sha(baseline),
        "row_diff_mask_sha256": tensor_sha(changed.to(torch.uint8)),
        "contract": "candidate differs from baseline iff L mod 4 is 1, 2, or 3; candidate row is [0,L) followed by -1",
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
