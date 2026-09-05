import copy

import pytest

from glm53_nvfp4.analyze_p8_smallm_graph import analyze


def payload(candidate=False, ms=1.0):
    graph = {"replays": 5, "timing_repeats": 100,
             "bitwise_deterministic": True, "matches_eager_output": True,
             "median_ms": ms}
    cell = {"tokens": 1, "cuda_graph": graph, "output_sha256": "out",
            "payload_sha256": {"x": "x", "topk_ids": "i",
                               "topk_weights": "w", "dense_reference": "r"}}
    return {"decision": "pass", "mode": "monolithic",
            "small_m_scheduler": candidate, "cells": [cell]}


def test_accepts_four_bit_exact_graph_cells():
    result = analyze([payload(ms=1.0) for _ in range(4)],
                     [payload(True, .2) for _ in range(4)])
    assert result["decision"] == "advance_to_integrated_tp4"
    assert result["minimum_reduction_percent"] == 80


def test_rejects_cross_arm_output_mismatch():
    baselines = [payload() for _ in range(4)]
    candidates = [payload(True, .2) for _ in range(4)]
    candidates[2] = copy.deepcopy(candidates[2])
    candidates[2]["cells"][0]["output_sha256"] = "different"
    with pytest.raises(ValueError, match="output differs"):
        analyze(baselines, candidates)
