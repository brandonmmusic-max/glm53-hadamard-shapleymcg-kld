FROM klc/glm53-flash-nvfp4:r19-sm120-tp4-ep4-dcp4-v80-mxfp6-h16-all

COPY b12x_h16/b12x/moe/_shared/kernels/dynamic.py /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/dynamic.py
COPY b12x_h16/b12x/moe/_shared/kernels/dynamic.py /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/dynamic.py
COPY p8_mcg/w4a8_mcg_decode.py /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a8_mcg_decode.py
COPY p8_mcg/w4a8_mcg_decode.py /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/w4a8_mcg_decode.py
COPY p8_mcg/patch_split_phases.py /tmp/patch_split_phases.py

RUN python3 /tmp/patch_split_phases.py \
      /opt/infernal-invocation/b12x/b12x \
      /opt/venv/lib/python3.12/site-packages/b12x \
    && rm /tmp/patch_split_phases.py

RUN find /opt/infernal-invocation/b12x/b12x /opt/venv/lib/python3.12/site-packages/b12x \
      -type f -name '*.pyc' -delete

LABEL org.klc.experiment="glm53-p8-k3-k4-mcg-e4m3-scaled-identity-boundary-v10" \
      org.klc.parent.digest="sha256:8e6856039c449aa202f113c3834af09e4e1dcb4c9c9b7e7eb009608f3772627a" \
      org.klc.codec="procedural-mcg-alpha2-to-e4m3-mxf8f6f4" \
      org.klc.weight-scale="physical-ue8m0-k32-consumed-by-mx-mma" \
      org.klc.ldlq="false" \
      org.klc.isa-cost="mxf8f6f4-is-2x-nvfp4-mma-issue"
