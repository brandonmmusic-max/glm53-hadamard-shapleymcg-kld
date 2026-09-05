FROM sha256:0336113e0fff876cccf9e6ac5347528ae59f4ad894a0ce7cb4c4e90b4651a745

# Fail closed before replacing any inherited source. The parent is the
# immutable P8 tail-V2 image, not a mutable local tag.
RUN test "$(sha256sum /usr/lib/python3.12/sitecustomize.py | cut -d ' ' -f 1)" = "43d81125d92376b1a69d53a71126a041cc9a18d8080e92dea0a2ae23be138b1e" \
    && test "$(sha256sum /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/dynamic.py | cut -d ' ' -f 1)" = "7211657f8508dfbf12d2f4158562144fe86741d1f72ea32cc1dc2d6c3dc6b4d6" \
    && test "$(sha256sum /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/dynamic.py | cut -d ' ' -f 1)" = "7211657f8508dfbf12d2f4158562144fe86741d1f72ea32cc1dc2d6c3dc6b4d6" \
    && test "$(sha256sum /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_small_m.py | cut -d ' ' -f 1)" = "a0c398e9d412672d1138c69a5c0c3677bb29b7215379c701062d29b8b9b9265f" \
    && test "$(sha256sum /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/p8_small_m.py | cut -d ' ' -f 1)" = "a0c398e9d412672d1138c69a5c0c3677bb29b7215379c701062d29b8b9b9265f" \
    && test "$(sha256sum /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_narrow_fc1.py | cut -d ' ' -f 1)" = "84d28a0a72b474ce3fde4eeb11a371bfb1c58e843928382992aa99457409f8d1" \
    && test "$(sha256sum /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/p8_narrow_fc1.py | cut -d ' ' -f 1)" = "84d28a0a72b474ce3fde4eeb11a371bfb1c58e843928382992aa99457409f8d1" \
    && test "$(sha256sum /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a8_phase1.py | cut -d ' ' -f 1)" = "7222df68bb6ad7ec5ca69cf7c4ffd5c3d6b1ddc240e6375c4b1e77bcc151909c" \
    && test "$(sha256sum /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/w4a8_phase1.py | cut -d ' ' -f 1)" = "7222df68bb6ad7ec5ca69cf7c4ffd5c3d6b1ddc240e6375c4b1e77bcc151909c" \
    && test "$(sha256sum /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a8_phase2.py | cut -d ' ' -f 1)" = "be317f7f76153ff5f60d2d1cb76f15dae14e6252a109b2d448e7dd8852f66194" \
    && test "$(sha256sum /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/w4a8_phase2.py | cut -d ' ' -f 1)" = "be317f7f76153ff5f60d2d1cb76f15dae14e6252a109b2d448e7dd8852f66194" \
    && test ! -e /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_h128_fc1.py \
    && test ! -e /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/p8_h128_fc1.py \
    && test ! -e /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_coupled_topk.py \
    && test ! -e /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/p8_coupled_topk.py \
    && test ! -e /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_coupled_prefill_fc1.py \
    && test ! -e /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/p8_coupled_prefill_fc1.py \
    && test ! -e /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_coupled_prefill_fc2.py \
    && test ! -e /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/p8_coupled_prefill_fc2.py \
    && test ! -e /opt/p8-coupled-runtime \
    && test "$(sha256sum /opt/infernal-invocation/vllm/vllm/models/glm5next/nvidia/ops/kpool_compress.py | cut -d ' ' -f 1)" = "494192195da43c46d99a684555fc10fd13a19e89288cb9f51da2536ccdf1f251"

COPY runtime_patch/p8_native_kernel.py /opt/p8-coupled-runtime/p8_native_kernel.py
COPY runtime_patch/p8_coupled_scales.py /opt/p8-coupled-runtime/p8_coupled_scales.py
COPY runtime_patch/p8_smallm_schedule.py /opt/p8-coupled-runtime/p8_smallm_schedule.py
COPY runtime_patch/p8_coupled_prefill_plan.py /opt/p8-coupled-runtime/p8_coupled_prefill_plan.py
COPY runtime_patch/sitecustomize.py /usr/lib/python3.12/sitecustomize.py

# PYTHONPATH selects the editable B12X tree first, but both installed copies
# are kept byte-identical to prevent a different launcher from seeing old code.
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/dynamic.py
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_h128_fc1.py /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_h128_fc1.py
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_narrow_fc1.py /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_narrow_fc1.py
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_small_m.py /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_small_m.py
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_coupled_topk.py /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_coupled_topk.py
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_coupled_prefill_fc1.py /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_coupled_prefill_fc1.py
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_coupled_prefill_fc2.py /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_coupled_prefill_fc2.py
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/dynamic.py
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_h128_fc1.py /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/p8_h128_fc1.py
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_narrow_fc1.py /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/p8_narrow_fc1.py
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_small_m.py /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/p8_small_m.py
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_coupled_topk.py /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/p8_coupled_topk.py
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_coupled_prefill_fc1.py /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/p8_coupled_prefill_fc1.py
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_coupled_prefill_fc2.py /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/p8_coupled_prefill_fc2.py

COPY runtime_patch/p8_coupled_image/image_manifest.json /opt/p8-coupled-runtime/image-manifest.json
COPY runtime_patch/p8_coupled_image/verify_image.py /opt/p8-coupled-runtime/verify_image.py

ENV PYTHONPATH=/opt/p8-coupled-runtime:/opt/exllamav3:/opt/infernal-invocation/vllm:/opt/infernal-invocation/b12x

ARG KLC_SOURCE_COMMIT
ARG KLC_SOURCE_TREE
RUN test -n "$KLC_SOURCE_COMMIT" \
    && test -n "$KLC_SOURCE_TREE" \
    && find /opt/infernal-invocation/b12x/b12x /opt/venv/lib/python3.12/site-packages/b12x /opt/p8-coupled-runtime \
        -type f -name '*.pyc' -delete \
    && /opt/venv/bin/python /opt/p8-coupled-runtime/verify_image.py

LABEL org.opencontainers.image.title="GLM-5.3 P8 coupled research runtime" \
      org.opencontainers.image.licenses="LicenseRef-ShapleyMCG-1.0" \
      org.klc.experiment="glm53-p8-coupled-h512-h128-suh-svh-v2" \
      org.klc.qualification="research-only-not-device-qualified" \
      org.klc.parent.digest="sha256:0336113e0fff876cccf9e6ac5347528ae59f4ad894a0ce7cb4c4e90b4651a745" \
      org.klc.tail-v2.sha256="494192195da43c46d99a684555fc10fd13a19e89288cb9f51da2536ccdf1f251" \
      org.klc.codec="procedural-mcg-alpha2-to-e4m3-mxf8f6f4" \
      org.klc.boundary="coupled-h512-h128-suh-svh-v1" \
      org.klc.ldlq="false" \
      org.klc.isa-cost="mxf8f6f4-is-2x-nvfp4-mma-issue" \
      org.klc.device-support="decode-m1-prefill-m64-n128-unqualified" \
      org.klc.source.commit="$KLC_SOURCE_COMMIT" \
      org.klc.source.tree="$KLC_SOURCE_TREE"
