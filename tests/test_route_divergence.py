import base64
import io

import numpy as np
import pytest

from glm53_nvfp4.analyze_route_divergence import spearman
from glm53_nvfp4.route_eval import decode_routed_experts


def _encode(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode()


def test_decode_routed_experts_slices_glm_layers_and_narrows_ids():
    routes = np.empty((5, 46, 8), dtype=np.int32)
    for slot in range(8):
        routes[:, :, slot] = slot
    decoded = decode_routed_experts(_encode(routes), expected_tokens=5)
    assert decoded.shape == (5, 42, 8)
    assert decoded.dtype == np.uint16
    assert np.array_equal(decoded, routes[:, 3:45].astype(np.uint16))


def test_decode_routed_experts_requires_one_row_per_prompt_token():
    routes = np.empty((5, 46, 8), dtype=np.int32)
    for slot in range(8):
        routes[:, :, slot] = slot
    with pytest.raises(RuntimeError, match="unexpected routed-expert geometry"):
        decode_routed_experts(_encode(routes), expected_tokens=4)


def test_decode_routed_experts_rejects_duplicate_topk():
    routes = np.zeros((5, 46, 8), dtype=np.int32)
    with pytest.raises(RuntimeError, match="duplicate"):
        decode_routed_experts(_encode(routes), expected_tokens=5)


def test_spearman_handles_ties_and_direction():
    x = np.asarray([1.0, 1.0, 2.0, 3.0])
    assert spearman(x, x) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert spearman(np.ones(4), x) == 0.0
