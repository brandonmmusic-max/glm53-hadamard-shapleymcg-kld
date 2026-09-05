import copy

import pytest

from glm53_nvfp4.analyze_p8_n256_device import analyze


def payload(ms):
    graph = {"replays": 5, "timing_repeats": 100,
             "bitwise_deterministic": True, "matches_eager_output": True,
             "median_ms": ms}
    cell = {"tokens": 1, "cuda_graph": graph, "output_sha256": "same",
            "payload_sha256": {"x": "x", "topk_ids": "i",
                               "topk_weights": "w", "dense_reference": "r"}}
    return {"decision": "pass", "cells": [cell]}


def test_stops_small_valid_gain():
    result = analyze([payload(1) for _ in range(4)], [payload(.98) for _ in range(4)])
    assert result["decision"] == "stop_n256_before_integration"


def test_rejects_output_mismatch():
    controls = [payload(1) for _ in range(4)]
    candidates = [payload(.5) for _ in range(4)]
    candidates[0] = copy.deepcopy(candidates[0])
    candidates[0]["cells"][0]["output_sha256"] = "different"
    with pytest.raises(ValueError, match="output mismatch"):
        analyze(controls, candidates)
