# V3 learned candidate chunk-mount amendment

Date: 2026-09-03. This amendment was made after learned conditional-fit failed a second time during checkpoint loading and before it produced any inference or KLD row. Selection remained unopened.

The complete learned checkpoint is a lightweight index whose 168 redirected chunk symlinks resolve into `/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-v3-large`. The runtime mounted the candidate directory and the model-storage campaign path, but not that root-filesystem chunk tree. Loading therefore failed with `FileNotFoundError: /model/learned-layer-003-experts-000-072.safetensors` after resolving the index entry inside the container.

The correction bind-mounts the existing chunk root read-only at the identical absolute path. No tensor, index, rotation, runtime arithmetic, role, metric, or decision rule changes. The runner is added to the rotation-freeze analysis files, and a new freeze receipt is created at a clean commit before learned conditional-fit resumes or selection wave 1 opens. The two pre-inference startup failures remain preserved.
