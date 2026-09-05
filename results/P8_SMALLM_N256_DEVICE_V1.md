# P8 small-M N256 FC2 device gate

Decision: **stop N256 before integrated TP4.**

| Physical GPU | N128x2 control | N256 A-reuse | Reduction |
|---:|---:|---:|---:|
| 0, Max-Q | 0.144128 ms | 0.142144 ms | 1.3766% |
| 1, Workstation | 0.117664 ms | 0.115552 ms | 1.7949% |
| 2, Max-Q | 0.145888 ms | 0.143696 ms | 1.5025% |
| 3, Workstation | 0.117696 ms | 0.115712 ms | 1.6857% |

The candidate compiled, captured and remained bit-exact on all four GPUs.
Payload hashes matched across arms; five graph replays matched eager and one
another. Each median contains 100 graph-timing samples.

The optimization is real but much too small. It falls far below the frozen 35%
gate sized to the remaining 2.27 ms/token product deficit, so no full-model run
is authorized for this arm. This negative result localizes the bottleneck:
duplicated activation/SFA staging across the two FC2 halves was not materially
responsible for the remaining decode gap.

The candidate still executes the procedural K4 MCG decoder in the kernel
prologue, feeds fully scaled E4M3 fragments with physical UE8M0/32 directly to
native `mxf8f6f4`, and preserves the monolithic per-K128 BF16 boundary. No
decoded-weight buffer or BF16 weight dequantization was introduced.

No protected role or teacher logits were opened. No LDLQ or BlockLDLQ was
used. Raw cells and the machine-readable decision are under
`evidence/opened/codec-v2/p8-smallm-n256-device-v1/`.
