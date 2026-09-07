# TrellisMX SM120 source-only runtime overlay

This subproject builds `trellismx-runtime-sm120`. It contains the v11
mixed-rate manifest and 15 SHA-256-pinned source files needed to reproduce the
separate immutable runtime image.

Installing the wheel does **not** make the runtime executable and imports no
CUDA, B12X, CUTLASS, or vLLM dependency. Device execution requires the pinned
image and separate authorization.

The root TrellisMX CLI can verify the installed source overlay:

```bash
CUDA_VISIBLE_DEVICES= trellismx runtime-info
```

License and attribution files are included in this package.
