# V9 eager device closure passed

Synthetic layer3/rank0 TP4-local, eight routed experts, fixed draw0 and Flash
capped SiLU; native coupled P8 MoE, E4M3 activations, UE8M0/32 weights,
4.25 payload bpw plus channel metadata (~4.25398 effective). Attention/KV: N/A.
P8 mxf8f6f4 has twice NVFP4's MMA issue count. No KLD/speed claim.

Image `sha256:ad6b26bf6d1f265d99b09383485ddef82a4acfaea43e28af46341ebb41da24e3`.
Runtime source `336e082785904e4c0c2bd887fe437ae14adad209`.

All four exact input/intermediate payload/scale gates passed on every repeat.
Each case produced one unique output hash across five eager repeats:

| Tokens | Final relative L2 versus reference |
| ---: | ---: |
| 1 | 0.0000932577604544349 |
| 2 | 0.000000008227113212910808 |
| 64 | 0.00003542305057635531 |
| 65 | 0.00003517053846735507 |

Receipts are in `evidence/preparation/p8-v9/`:
M1 result SHA256 `2e3b1d09940259ee5792c4464f0187cc9a44e386e72686b046d9284dadd299de`;
prefill result SHA256 `d8a0885d7fda130b959bacced2c0af13901ec96c0c6d8db844100b5c257bd262`.
Raw cosine calculations occasionally exceed one slightly due to FP32 reduction;
they are preserved, not interpreted as better-than-perfect agreement.

This verifies this synthetic eager fixture only. CUDA graphs, real packed
layers/ranks, integrated serving, teacher KLD, throughput, and all42 remain
unqualified. Earlier failures remain archived. Production stays off.
