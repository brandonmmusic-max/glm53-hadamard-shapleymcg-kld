import torch

from glm53_nvfp4.block_rotation import apply_weight_rotation
from glm53_nvfp4.make_rotation_canary import signed_permutation16
from glm53_nvfp4.modelopt import PackedNVFP4, dequantize, quantize
from glm53_nvfp4.permute_packed_chunk import transform


def test_exact_packed_permutation_preserves_scales_and_rotates_codes():
    generator = torch.Generator().manual_seed(66)
    weight = torch.randn(32, 32, generator=generator)
    packed = quantize(weight)
    tensors = {
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight": packed.weight,
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale": packed.weight_scale,
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale_2": packed.weight_scale_2,
    }
    result, changed = transform(tensors, layer=3, scope="gate-up")
    assert changed == ["model.language_model.layers.3.mlp.experts.0.gate_proj.weight"]
    transformed = PackedNVFP4(
        result[changed[0]],
        result[changed[0].removesuffix(".weight") + ".weight_scale"],
        result[changed[0].removesuffix(".weight") + ".weight_scale_2"],
    )
    torch.testing.assert_close(
        dequantize(transformed),
        apply_weight_rotation(dequantize(packed), signed_permutation16()),
        atol=0,
        rtol=0,
    )
    assert torch.equal(
        transformed.weight_scale.view(torch.uint8),
        packed.weight_scale.view(torch.uint8),
    )
    assert torch.equal(transformed.weight_scale_2, packed.weight_scale_2)


def test_negative_identity_is_raw_sign_xor_and_an_involution():
    # Include +0 and -0 explicitly in both low and high nibbles.
    weight = torch.tensor([[0x80, 0x08, 0x10, 0x01, 0xF7, 0x7F, 0x88, 0x00]], dtype=torch.uint8)
    tensors = {
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight": weight,
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale": torch.ones((1, 1), dtype=torch.float8_e4m3fn),
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale_2": torch.tensor(1.0),
    }
    once, _ = transform(
        tensors, layer=3, scope="gate-up", kind="negative-identity"
    )
    key = "model.language_model.layers.3.mlp.experts.0.gate_proj.weight"
    torch.testing.assert_close(once[key], weight ^ 0x88, atol=0, rtol=0)
    twice, _ = transform(
        once, layer=3, scope="gate-up", kind="negative-identity"
    )
    torch.testing.assert_close(twice[key], weight, atol=0, rtol=0)
