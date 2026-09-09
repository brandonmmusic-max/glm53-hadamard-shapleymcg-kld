# Selected reference runtime

The published tested image is `verdictai/trellismx:glm53-flash-p8-r27-reference-20260909@sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf`. Use ../compose.yaml or ../serve.sh for24slots and measured collective settings.

This Dockerfile reconstructs the selected Python source overlay on the pinned September8 public base. source-manifest.json records all33changed Python files. It does not promise byte-identical image layers or GPU validation of a fresh rebuild. The baked launcher defaults to16slots; the public compose/wrapper explicitly select24. Override the old default entrypoint with /bin/bash /release/serve-r27-production.sh.
