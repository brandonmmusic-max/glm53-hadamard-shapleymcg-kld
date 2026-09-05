# Forced-M1 capture v3: TCP lifecycle preflight amendment

Status before v3 execution: the warmup repair worked on all four GPUs and v2
completed the entire N128 canary. The attempt then stopped at N64's port
preflight, before creating its container. No paired closure or new KLD exists.

## Measured v2 outcome

Plan SHA-256:
`0faf9049d5f565e887570a6eff6ceac431ad4b6d767f2a72ab4b324b348d29c3`.
N128 completed 2,047 rows by 154,880 vocabulary entries, with original
`torch.bfloat16` logits stored as FP32. The raw capture is 1,268,157,440 bytes,
SHA-256 `051ee7210a3321ec66c239022a7af9e0bd18ae54cf9d1bf2ed4775b7289932b9`.
All four warmup scopes closed correctly; all 168 layer/rank M1 dispatches,
returned forced tokens, final capture metadata and cleanup passed. The N128
stage exited 0, receipt SHA-256
`622d5b5e7ef2787e378f0c191663012c48fb0816bf4c1b68b904ba9c7556fbeb`.
The successful capture proves instrumentation on this one arm, not agreement
with N64, full-panel quality, or speed.

The [v2 terminal snapshot](../evidence/opened/codec-v2/p8-forced-m1-v2-v2-stop/snapshot.json)
has SHA-256 `8077d19f119c2bbb79e609b79ca484dfdb6ad04d41e06cc700f614ce93d59e6f`.
It retains safe receipts and metadata, not raw logits or private environment.
An independent verifier replayed the original source's request/response,
capture and runtime checks; all 19 stage artifacts authenticated.

The N64 stage recorded `[Errno 98] Address already in use` before container
creation. The overall unit exited 1; no N64 or full-stage captures were made.
Services restored to backend-active/timer-inactive with no owned containers.
The original v2 worktree, image, plan and raw data remain unchanged.

## Diagnosis and correction before another measurement

Read-only diagnostics immediately after the failure found no listening socket
on 8023, one `TIME_WAIT` connection with local endpoint `127.0.0.1:8023`, and
a successful temporary bind with `SO_REUSEADDR`. The failed code checked the
same port using a plain bind immediately after shutting down the previous
server. These observations support a TCP lifecycle preflight false positive,
not a second live serving process or a model error.

V3 changes only the non-listening port probe to set `SO_REUSEADDR` before
binding. It never sets `SO_REUSEPORT`, never listens, and closes before launch.
A real localhost listener, including one using `SO_REUSEADDR`, still fails
the probe. CPU tests exercise that negative case, an actively closed local
connection, and the exact socket-option/call sequence. This does not eliminate
the ordinary race between any availability probe and a later server bind;
the serialized model-stack lock and container identity checks remain intact.

Use a new v3 plan, source commit, output and service. Reuse the exact v2 image
`sha256:0f1eae9329965d68713857e4a5a12e9c5440c866b532e7ba288dc2ae4067fad9`;
there is no image, model, encoder, kernel arithmetic or warmup change.
Run the same four-stage N128/N64 canary then N128/N64 full-panel sequence.
The prior N128 capture is retained, not substituted into the new pair.
Exact 2,047-row canary agreement is still required before opening either full
stage; all32 approved conditional-fit windows remain balanced eight/domain.
No tolerance, stopping, KLD estimator or role change, and no LDLQ.

The new [v3 plan](../experiments/p8-forced-m1-v2-v3.json) is sealed with
SHA-256 `7b675117a4a2c8a5abe87056dae6a3b6bcd1db2d44d8d0b5c45ad0f33b219085`.
All 32 approved teacher files were freshly byte-verified before sealing.
The service is `glm53-p8-forced-m1-v2-v3.service`, writing a fresh
`forced-m1-v2-v3` directory beside the preserved earlier attempts. Launch
does not imply canary agreement or protocol completion.

P8 remains E4M3 `mxf8f6f4`, with twice NVFP4's MMA issue count. Capture timing
is not a speed result. Allocation remains stopped; protected selection,
confirmation, final and the 28 reserved confirmation logits stay unopened.
