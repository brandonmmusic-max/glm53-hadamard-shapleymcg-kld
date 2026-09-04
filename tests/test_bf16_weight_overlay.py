import pytest

from glm53_nvfp4.bf16_weight_overlay import PROJECTIONS, _projection_for


@pytest.mark.parametrize("projection", sorted(PROJECTIONS))
def test_projection_for_exact_expert_weight_suffix(projection):
    name = f"model.language_model.layers.22.mlp.experts.7.{projection}.weight"
    assert _projection_for(name) == projection


def test_projection_for_rejects_quantization_side_tensor():
    assert _projection_for(
        "model.language_model.layers.22.mlp.experts.7.gate_proj.weight_scale"
    ) is None
