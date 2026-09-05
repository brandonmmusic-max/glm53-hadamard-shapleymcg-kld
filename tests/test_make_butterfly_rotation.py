import pytest
import torch

from glm53_nvfp4.make_butterfly_rotation import build
from glm53_nvfp4.quantize_shared_mid_butterfly_layer import _canonical_rotation


def test_build_butterfly_rotation_identity_and_table_size():
    rotation, receipt = build(3, 0.0)
    torch.testing.assert_close(rotation, torch.eye(16), atol=0, rtol=0)
    assert receipt["table_bytes"] == 1024
    assert receipt["ldlq"] is False


def test_build_butterfly_rotation_rejects_out_of_range():
    with pytest.raises(ValueError, match="angle_pi"):
        build(3, 0.251)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_canonical_payload_check_keeps_cpu_builder_bits_on_cuda():
    rotation, _ = build(3, -0.25)
    expected = _canonical_rotation(-0.25, "cuda")
    assert torch.equal(rotation.cuda(), expected)
