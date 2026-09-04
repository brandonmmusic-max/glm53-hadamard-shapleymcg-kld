import pytest

from glm53_nvfp4.validate_layer import ratio_fails


def test_gptq_requires_strict_improvement_over_matched_rtn():
    assert not ratio_fails("gptq", 0.999)
    assert ratio_fails("gptq", 1.0)
    assert not ratio_fails("mr-gptq", 0.999)
    assert ratio_fails("mr-gptq", 1.0)
    assert not ratio_fails("fpquant-mr-gptq", 0.999)
    assert ratio_fails("fpquant-mr-gptq", 1.0)
    assert not ratio_fails("trellis-nvfp4", 0.999)
    assert ratio_fails("trellis-nvfp4", 1.0)
    assert not ratio_fails("identity-k4", 0.999)
    assert ratio_fails("identity-k4", 1.0)


def test_rtn_requires_identity_with_matched_rtn_control():
    assert not ratio_fails("rtn", 1.0)
    assert not ratio_fails("rtn", 1.0 + 1e-9)
    assert ratio_fails("rtn", 1.01)
    with pytest.raises(ValueError):
        ratio_fails("unknown", 1.0)
