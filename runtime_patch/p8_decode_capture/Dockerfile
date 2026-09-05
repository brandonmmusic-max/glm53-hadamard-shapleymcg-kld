# Build through scripts/build_p8_decode_capture_image.py with the legacy local
# builder so this image ID is resolved locally, without registry/tag lookup.
FROM sha256:6c08dffb4184c2704173a12909f4bbfaaa866351e55cbf03a2182741baf81141
# Parent: sha256:6c08dffb4184c2704173a12909f4bbfaaa866351e55cbf03a2182741baf81141
# This image is for numerical capture, never throughput measurement.
COPY __init__.py processor.py v2_hook.py /opt/venv/lib/python3.12/site-packages/p8_decode_capture/
COPY sampler-v2-hook.patch /opt/p8-decode-capture/sampler-v2-hook.patch
RUN test "$(sha256sum /opt/infernal-invocation/vllm/vllm/v1/worker/gpu/sample/sampler.py | cut -d ' ' -f 1)" = "0b56a1c80e2823235fc7df202c6784fd11b83ecaf80af745dceef5a143307711" \
    && test "$(sha256sum /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/sampler.py | cut -d ' ' -f 1)" = "0b56a1c80e2823235fc7df202c6784fd11b83ecaf80af745dceef5a143307711" \
    && cd /opt/infernal-invocation/vllm \
    && git apply --check /opt/p8-decode-capture/sampler-v2-hook.patch \
    && git apply /opt/p8-decode-capture/sampler-v2-hook.patch \
    && cp vllm/v1/worker/gpu/sample/sampler.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/sampler.py \
    && cmp vllm/v1/worker/gpu/sample/sampler.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/sampler.py \
    && test "$(sha256sum /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_small_m.py | cut -d ' ' -f 1)" = "a0c398e9d412672d1138c69a5c0c3677bb29b7215379c701062d29b8b9b9265f" \
    && /opt/venv/bin/python -m py_compile /opt/venv/lib/python3.12/site-packages/p8_decode_capture/processor.py /opt/venv/lib/python3.12/site-packages/p8_decode_capture/v2_hook.py vllm/v1/worker/gpu/sample/sampler.py
