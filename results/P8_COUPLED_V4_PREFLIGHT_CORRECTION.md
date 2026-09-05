# V4 source identity correction

2026-09-05: initial v4 preflight at source `93a5d9a` failed before staging
or building. The recipe copied an incorrect full expansion of short commit
`b1745f4` from an agent message. `git rev-parse b1745f4` establishes the actual
object as `b1745f43e688ed25e9278488a93e684979086594`.

Decision 1: correct the builder, manifest and test to this verified object.
Keep every runtime source hash, parent image, gate and fixture unchanged.
The ancestry preflight correctly rejected the invalid object:
`fatal: Not a valid commit name b1745f44c72fdd607733d892c6cbe58a1b61e8cb`.
No Docker build, GPU use, output-directory creation or service change occurred
in the rejected attempt. Retry only after verifying the resolved ancestry.
