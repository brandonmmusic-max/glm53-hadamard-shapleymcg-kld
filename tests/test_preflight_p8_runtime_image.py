import pytest
from glm53_nvfp4.preflight_p8_runtime_image import verify_constructor


def test_accepts_explicit_p8_abi():
    verify_constructor('class MoEDynamicKernelBackend:\n def __init__(self, *, trellis_codebook=None, trellis_scaled=False, trellis_identity_boundary=False, deterministic_output=False): pass')


def test_rejects_base_image_abi():
    with pytest.raises(ValueError, match="missing"):
        verify_constructor('class MoEDynamicKernelBackend:\n def __init__(self, deterministic_output=False, **kwargs): pass')
