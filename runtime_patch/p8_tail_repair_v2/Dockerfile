FROM sha256:0f1eae9329965d68713857e4a5a12e9c5440c866b532e7ba288dc2ae4067fad9

COPY kpool-tail-after-valid-history-v2.patch /opt/p8-tail-repair-v2/kpool-tail-after-valid-history-v2.patch
RUN test "$(sha256sum /opt/infernal-invocation/vllm/vllm/models/glm5next/nvidia/ops/kpool_compress.py | cut -d ' ' -f 1)" = "01bfba91f667214760e1fe8af8c4151498ae9b0407493a4670e255945ae01a58" \
    && cd /opt/infernal-invocation/vllm \
    && git apply --check /opt/p8-tail-repair-v2/kpool-tail-after-valid-history-v2.patch \
    && git apply /opt/p8-tail-repair-v2/kpool-tail-after-valid-history-v2.patch \
    && /opt/venv/bin/python -m py_compile vllm/models/glm5next/nvidia/ops/kpool_compress.py \
    && grep -q 'P8 tail-after-valid-history-v2' vllm/models/glm5next/nvidia/ops/kpool_compress.py
