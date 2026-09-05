# The local tag is preflighted to the immutable base image ID before build.
# BuildKit otherwise attempts remote registry resolution for the local digest.
FROM klc/glm53-flash-nvfp4:p8-mcg-identity-v14
# Build context is the campaign worktree. No model or role data is copied.
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/dynamic.py
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_small_m.py /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/p8_small_m.py
# The image's Python path resolves the editable B12X tree before site-packages.
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/dynamic.py
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_small_m.py /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_small_m.py
