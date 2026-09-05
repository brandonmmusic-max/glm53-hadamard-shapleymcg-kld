# Three-layer coupled P8 storage preflight

Prepared 2026-09-05 before candidate encoding. This is an arithmetic bound,
not an observed peak-use receipt. Layers: 3, 20, 22; E=288, H=4096,
I=2048; three routed projections per expert; TP4.

| Artifact | Bytes |
|---|---:|
| One layer K4 + UE8M0/32 payload | 3,850,371,072 |
| Three-layer payload | 11,551,113,216 |
| Three-layer FP16 scale arrays across TP4 | 10,813,440 |
| Three-layer generated runtime sign arrays across TP4 | 36,864 |
| Two payload copies plus two scale copies and runtime signs | 23,123,890,176 |
| Dense BF16 copy of three layers | 43,486,543,872 |
| User's aggregate new-NVMe ceiling | 30,000,000,000 |

Formula: each layer contains `288 * 4096 * 2048 * 3 = 7,247,757,312`
routed weights. Payload bytes equal this count times `4.25 / 8`.
The scale layout per rank is two H-wide shared FP16 arrays and three
E-by-I/4 private FP16 arrays: 901,120 bytes/rank/layer.

Run the encoder without `--dense-output`. The nominal two-copy budget leaves
6,876,109,824 bytes for headers, chunk repetition of shared scales, serialized
codec tables, preparation artifacts, image/cache growth, receipts and captures.
These omitted quantities must be measured or conservatively bounded before
launch; the arithmetic alone does not authorize allocation. Existing baseline
campaign growth also counts against the user's aggregate ceiling.

Before encoding and repacking, measure actual new bytes and available NVMe
space. Include compressed chunks and final sidecars simultaneously in peak-use
accounting. After artifact hashes and closure receipts are durable, retire only
explicitly identified redundant artifacts if needed. Do not create a dense
three-layer BF16 overlay under this budget.

The scale-only increment is 0.00397858796 bpw. The current encoder sidecar's
288 one-byte draw IDs raise the stored tensor increment to 0.00397985953 bpw.
The runtime regenerates 3,072 FP16 sign bytes per rank/layer rather than
serializing them; charging that resident array too gives 0.00399342290 bpw and
4.25399342290 total, before safetensors headers. File overhead remains separate. Native P8 uses
E4M3 weights/activations and UE8M0/32 scales; its mxf8f6f4 instruction requires
twice the MMA issue count of NVFP4. No device speed is inferred here.
