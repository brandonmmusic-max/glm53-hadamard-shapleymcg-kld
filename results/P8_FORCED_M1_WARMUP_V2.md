# Forced-M1 capture: preserved startup failure and v2 amendment

Status: the v1 capture attempt failed during startup, before any evaluation
request or logit capture. This is not a codec/KLD result. The v2 instrumentation
amendment is built and CPU-reviewed; no new GPU correctness result is claimed.

## Preserved v1 evidence

The sealed plan was `experiments/p8-forced-m1-v2-v1.json`, SHA-256
`e3332469bc2ed9aec86eabb86b8dc57f76123542e1de309b9136893e4014725a`.
Only `canary-n128` started, using capture image
`sha256:071da1f9e9b24709e59a0959b97a8d524e82fca2fbf4de9a5dc2113a938c6b60`.
The root receipt records exit 1, successful identity checks, no live owned
containers, backend restored active, and timer inactive as found. Independent
verification authenticated the root-to-stage receipt and all ten stage files.
Both requests and captures directories were empty; neither full stage started.
Teacher files were hash-verified during planning: zero evaluation captures does
not mean the conditional-fit role was never opened.

The pinned vLLM `warmup_kernels` calls the model with a registered synthetic
two-token prompt. The capture helper accepted only one-token real prompts and
separately recognized an unregistered all-padding dummy batch. Neither rule
covered this startup request. The preserved error is
`V2 decode capture requires one prompt token at request row zero`.

Portable evidence: [failure snapshot](../evidence/opened/codec-v2/p8-forced-m1-v2-v1-failure/snapshot.json),
SHA-256 `cf4c013dd12267632e25cb0ddb8816690070610799801091e21d1fc9c002942a`.
Private logs and environment remain local and are represented by hashes;
canonical failure frames and selected hardware/thermal fields are sanitized.
The original worktree, image, plan, and raw output remain unchanged.

## Amendment, before new measurements

Create a new image, source commit, plan/seal, output directory, and service
with suffix `v2-v2`. Retain the v1 numerical protocol unchanged: N128 then N64
on one complete 2,047-row canary, exact-byte agreement before either full
stage, then all 32 conditional-fit windows (eight per domain, 65,504 causal
rows), paired KLD analysis and preserved failures. No LDLQ, model re-encoding,
kernel arithmetic change, role substitution, or tolerance relaxation.

The sole new runtime permission is a lexical scope entered by the pinned
startup warmup function. It recognizes exactly one registered synthetic
request and two sampler calls, validates mappings and parameters, returns
those warmup logits untouched, then requires cleanup and closes the scope.
Every real request outside the scope retains the existing capture guards.
Request-name spoofing alone does not grant bypass. A rank-tagged closure
marker is required for all four ranks before the launcher sends evaluation
requests and again in the terminal runtime audit.

The builder must verify and patch both installed and editable vLLM copies.
Original `warmup.py` SHA-256:
`696cdd462f58908f3511e9983fe18aa3811acecdd6da1ff580bbab99aed08dcc`.
Because `/runtime-patch` precedes site-packages on PYTHONPATH, the new launch
must bind the sealed v2 worktree instead of silently importing the old helper.
All inherited P8 arithmetic sources remain frozen, including the executed
FC2 source rather than an unexecuted local donor copy.

## Built v2 image and CPU validation

Dedicated image:
`sha256:0f1eae9329965d68713857e4a5a12e9c5440c866b532e7ba288dc2ae4067fad9`.
[Build receipt](../evidence/opened/codec-v2/p8-decode-capture-image-v2/receipt.json)
SHA-256 `fdd57fa758d33f81d88706487232e2f68b89a24f5c1e398e83516e212f14ffd6`.
The build performed no GPU work, used the local digest-pinned speed image,
and authenticated both sampler copies, both patched warmup copies, the three
capture-package sources, and inherited FC2. Patched warmup SHA-256 is
`de321498f305e2f61d5cfe4701d19a066ebfec83147f527b2ec82bfb653ea8c7`.

The combined focused CPU suite passed 155 tests, including capture/forced-token
semantics, warmup negative cases, launcher/image receipts, role/row protocol,
paired analysis, historical cold replay and startup-failure preservation.
The independently reviewed warmup implementation rejects nested/repeated
scopes and retains a terminal failure latch. A CPU-only check inside the
exact built image passed the real `SamplingParams.for_sampler_warmup()`
signature check and patched-function import; it cannot instantiate the
GPU-backed request state or establish runtime closure.

The v2 launcher replays the historical speed verifier in its original,
unchanged worktree using `python -I -B`, because that verifier authenticates
absolute historical prerequisite paths. Original source hashes are checked
before and after replay; its plan is not rewritten to fit the new worktree.

## Boundaries

The new plan is [p8-forced-m1-v2-v2.json](../experiments/p8-forced-m1-v2-v2.json),
SHA-256 `0faf9049d5f565e887570a6eff6ceac431ad4b6d767f2a72ab4b324b348d29c3`.
Planning freshly byte-verified all 32 approved teacher files, confirmed eight
windows in each of four domains, pinned 29 source files, and budgeted
83,698,391,040 raw capture bytes plus 20 GiB free-space margin. The service
name is `glm53-p8-forced-m1-v2-v2.service`; the fresh raw output is
`/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-smallm-scheduler-v1/forced-m1-v2-v2`.
Sealing and launch are not completion evidence; terminal receipts govern.

CPU tests and a successful image build are not capture proof. Capture timing
is not a throughput measurement. The separate five-cold-run speed pass remains
100.5501 versus 91.2200 decode tok/s and 6,959 versus 6,239 server-prefill
tok/s, between different EP/DCP serving regimes. P8 uses E4M3 `mxf8f6f4`
with twice NVFP4's MMA issue count; it is not the native P4 speed-class endpoint.
Allocation remains stopped pending numerical closure. Protected selection,
confirmation, final, and the 28 reserved confirmation logits stay unopened.

The current N64 runtime still needs a fresh detailed profile. A reviewer's
initial N32 grid-expansion suggestion was withdrawn after the installed
launcher was traced: it already doubles `mac=64` to 128 CTAs. Do not present
that as an unimplemented optimization or infer current bottleneck percentages
from the older N128 trace.
