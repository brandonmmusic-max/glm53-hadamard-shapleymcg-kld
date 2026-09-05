# The local tag is preflighted to the immutable base image ID before build.
# BuildKit otherwise attempts remote registry resolution for the local digest.
FROM klc/glm53-p8-smallm:v1-src-7a0d202
# Required local base ID:
# sha256:c9eddeca0d9dedf210eaaff918f1bf44d5338202902143f7d646d2325ad475ed
# Verify inherited FC2 itself before copying any candidate source.
RUN test "$(sha256sum /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_small_m.py | cut -d ' ' -f 1)" = "a0c398e9d412672d1138c69a5c0c3677bb29b7215379c701062d29b8b9b9265f"
# Build context is the campaign worktree. No model or role data is copied.
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/dynamic.py
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_narrow_fc1.py /opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/p8_narrow_fc1.py
# The image's Python path resolves the editable B12X tree before site-packages.
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/dynamic.py /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/dynamic.py
COPY runtime_patch/b12x_h16/b12x/moe/_shared/kernels/p8_narrow_fc1.py /opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/p8_narrow_fc1.py
