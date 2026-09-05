FROM sha256:0336113e0fff876cccf9e6ac5347528ae59f4ad894a0ce7cb4c4e90b4651a745

# Diagnostic-only image.  The parent is the validated tail-V2 image; neither
# the B12X writer nor its 288-byte record is replaced here.
COPY runtime_patch/nvfp4_mla_scale_counter.py /opt/nvfp4-mla-counter/nvfp4_mla_scale_counter.py
COPY runtime_patch/sitecustomize.py /opt/nvfp4-mla-counter/sitecustomize.py

ENV PYTHONPATH=/opt/nvfp4-mla-counter:${PYTHONPATH}

RUN test "$(sha256sum /opt/infernal-invocation/vllm/vllm/model_executor/layers/attention/mla_attention.py | cut -d ' ' -f 1)" = "a070481091d425eddc311e25c0cb770e7ed79d1ee3d196157ad28d789fb5c60c" \
    && test "$(sha256sum /opt/infernal-invocation/vllm/vllm/v1/attention/backends/mla/b12x_mla_sparse.py | cut -d ' ' -f 1)" = "0499c674b6890266b50fa0d5724dcfbb83cba3917714a6787e5dddc6feb65572" \
    && test "$(sha256sum /opt/infernal-invocation/b12x/b12x/attention/_shared/mla/kv_cache.py | cut -d ' ' -f 1)" = "7958acb694a8d7600dd5392b8899a42f4e01ca76a937dc82015cf77ad96feef9" \
    && /opt/venv/bin/python -m py_compile \
       /opt/nvfp4-mla-counter/nvfp4_mla_scale_counter.py \
       /opt/nvfp4-mla-counter/sitecustomize.py

LABEL org.klc.diagnostic="nvfp4-mla-static-scale-counter-v1" \
      org.klc.kv-abi="nope-288-byte-e2m1-e4m3-static-outer-scale" \
      org.klc.writer-mutation="none" \
      org.klc.timing-admissible="false" \
      org.klc.parent.tail-v2="sha256:0336113e0fff876cccf9e6ac5347528ae59f4ad894a0ce7cb4c4e90b4651a745"

