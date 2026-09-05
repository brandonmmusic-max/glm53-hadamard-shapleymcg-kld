"""B12X P4 trellis backend selected by the GLM routed-expert adapter.

This is a separate CUDA backend from the CuTe P8 dynamic backend. It consumes
the same vLLM modular router output but owns both P4 projections and their
activation prologues. ExLlamaV3/KQuant/QSRT/w4a8_trellis ports are identified
in runtime_patch/p4/p4_decode.cuh and THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

try:  # Runtime image mounts /runtime-patch directly on PYTHONPATH.
    from p4_native_kernel import P4NativeTPMoE
except ModuleNotFoundError as error:
    if error.name != "p4_native_kernel":
        raise
    from runtime_patch.p4_native_kernel import P4NativeTPMoE


class P4TrellisMoEBackend(P4NativeTPMoE):
    """Actual GLM serving entry: routed FC1 -> SwiGLU -> routed FC2.

    All matrix operations are the inherited native E2M1 K64 CUDA kernels.
    The enclosing vLLM runner owns shared experts and the TP all-reduce.
    """

    backend_name = "b12x-p4-trellis-mxf4nvf4"
    output_is_reduced = False

    def __init__(self, *args, **kwargs):
        kwargs["expected_geometry"] = (288, 4096, 512)
        super().__init__(*args, **kwargs)
        if (self.experts, self.hidden, self.intermediate, self.topk, self.limit) != (
            288, 4096, 512, 8, 10.0
        ):
            raise ValueError("GLM P4 serving requires E288 H4096 I512 TP4 topk8 SwiGLU10")
        self.prepare()

    def apply(self, x, topk_weights, topk_ids):
        return self(x, topk_weights, topk_ids)
