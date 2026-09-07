# Toolkit integration and preserved versions

## Sources and scope

Integrated the three toolkit commits by Brandon M. Music:

- `a73f0b85f781479aa138432d43fedac060b4fdbf`: installable toolkit and runtime-source package.
- `06b7cd1460b01c61feb85139b4a5973dbec47f8e`: CPU tooling expansion.
- `87071bc2dc931bfede7b8ac47beaaffd5ed5f592`: generic adapter/evaluation contracts.

Their source base is `6c7664a96809c2f4af1fedf6c26b7152e47fac0a`.
Integration base is default `main` at
`b15c68085e1b13d43da4168c9a6a3aa13bf424d4`, including the prominently
displayed 0.034181 coupled-checkpoint result and receipts.

This is a focused import of the three toolkit commits, not a blanket merge
of the 25 earlier divergent runtime/campaign commits. The research worktree,
checkpoint, production services and GPU configuration are untouched.

## Reconciled intent

| Surface | Resolution |
| --- | --- |
| README | Add install/extension instructions; retain the latest KLD headline and older experimental history instead of replacing them with the toolkit branch's rewritten README. |
| CITATION.cff | Adopt toolkit name/date while preserving main's complete references and human attribution. |
| Python packaging | Install as `trellismx`; retain `glm53_nvfp4` and `bmxfp4` import packages. Add an explicit pytest test extra so CPU CI installs its runner. |
| Runtime overlay | Retain the toolkit's pinned source snapshot without overwriting current research runtime files. Validate snapshot hashes against its manifest, not against unrelated current files. |
| Encoder access | Include the requested toolkit and research encoder contract; do not claim arbitrary-model conversion or new CUDA validation. |

These are compatible intents. There is no numerical algorithm change in
this integration and no new KLD or GPU speed claim.

## What is implemented versus unfinished

Implemented surfaces include a CLI, K3/K4/K5 CPU reference decoding,
P4/P8 containers and fixtures, sidecar validation, GLM index/config
inspection, checkpoint plans, resumable atomic orchestration, role and
provenance validation, adapter contracts, and a source-only SM120 overlay.

Only GLM-5.3-Flash has a bundled architecture adapter. The main coupled
encoder is CUDA research source; ordinary CLI execution does not provide
an arbitrary-model BF16-to-P8 conversion service. A downstream model needs
its own tensor mapping, calibration/transform integration, kernels and
end-to-end evaluation. CPU tests do not establish device execution, speed,
or another model's quality. Future evaluation references are not evidence
that those evaluations were used for the recorded CF32 results.

## Older versions

No older branch, tag, model, results directory or legacy import package is
deleted. The previous main remains addressable at `b15c68085e1b13d43da4168c9a6a3aa13bf424d4`;
the original toolkit branch remains at `87071bc2dc931bfede7b8ac47beaaffd5ed5f592`;
the original campaign branch remains at `6c7664a96809c2f4af1fedf6c26b7152e47fac0a`.
Use a separate checkout of these revisions when reproducing an older
version; do not replace a working serving installation merely to install
the new CPU toolkit. ShapleyMCG and all applicable third-party licenses
remain in place. Repository visibility remains private.

## Validation

Validation is CPU-only with `CUDA_VISIBLE_DEVICES` empty:

- `python3 -m pytest -q tests/test_p4_codec.py tests/test_trellismx_*.py`: **85 passed**.
- Both wheels built with `pip wheel --no-deps --no-build-isolation`.
- Both installed with `--no-index --no-deps` in an isolated environment,
  using pre-existing system dependencies; no dependency downloads.
- Installed `trellismx capabilities` succeeded outside the source directory.
- Installed `trellismx runtime-info`: **passed**, 15 source files, no hash
  mismatches, `origin=installed-overlay`, `cuda_used=false`, `executable=false`.

No GPU run, full-model reencoding or runtime qualification is part of this
integration. Broad historical research tests were not rerun; the full
focused toolkit and portable codec suite above was run on the composed tree.
