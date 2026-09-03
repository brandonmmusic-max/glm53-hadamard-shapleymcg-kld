FROM klc/glm53-flash-nvfp4:r19-sm120-tp4-ep4-dcp4-v79-dflash2-packed-aux-candidate

COPY b12x/integration/vllm/plugin.py /opt/infernal-invocation/b12x/b12x/integration/vllm/plugin.py
COPY b12x/integration/vllm/fp6_serving.py /opt/infernal-invocation/b12x/b12x/integration/vllm/fp6_serving.py
COPY b12x/moe/_shared/kernels/dynamic.py /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/dynamic.py
COPY b12x/integration/vllm/plugin.py /opt/venv/lib/python3.12/site-packages/b12x/integration/vllm/plugin.py
COPY b12x/integration/vllm/fp6_serving.py /opt/venv/lib/python3.12/site-packages/b12x/integration/vllm/fp6_serving.py
COPY b12x/moe/_shared/kernels/dynamic.py /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/dynamic.py

RUN find /opt/infernal-invocation/b12x/b12x /opt/venv/lib/python3.12/site-packages/b12x \
      -type f -name '*.pyc' -delete

LABEL org.klc.experiment="glm53-mxfp6-h16-all-projection-v10" \
      org.klc.parent.digest="sha256:ed027a3a2ff93b9cf60c95f7adfaf676cabc8e040a28cffa7486a262c82fdfbe" \
      org.klc.rotation="fixed-block16-sylvester-h16-input-and-post-swiglu"
