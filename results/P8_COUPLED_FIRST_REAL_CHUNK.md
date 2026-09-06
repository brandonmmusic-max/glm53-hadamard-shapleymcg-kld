# First coupled BF16 encoding chunk complete

Layer3 experts0:72, fixed V3 transform and 256 fit samples per expert,
domain-balanced calibration. Attention backend/KV: N/A (encoder). Target MoE:
coupled P8 native MCG/E4M3/UE8M0, E4M3 activations. Payload4.25 bpw plus
0.003978906 chunk metadata bpw; no serving or KLD measurement here.
P8 mxf8f6f4 requires twice NVFP4's MMA issue count.

- Elapsed311.4362 seconds; peak allocated CUDA1,567,717,376 bytes.
- Output963,555,616 bytes, SHA256
  `c062c971e61adb2156b2cd049851d1ff39e79fa0852794133646976dd82cb941`.
- Receipt SHA256`84857a0e5140c7ef3a686b6020c56f5f49ed84fc8fab8170d5dae1baf9d2e362`.
- 216 projection records: mean transformed-weight NMSE gate0.0064113254,
  up0.0064116440, down0.0065993236.

The full file hash and size were checked against the receipt before starting
the next fixed range. These NMSE values have no matched control in this report
and cannot establish a KLD benefit. No expert was selected or excluded based
on its result. The remaining layer3 sequence is sealed separately; packing,
real loader closure and the three-layer CF32 test remain outstanding.

Receipts: `evidence/preparation/p8-first-real-chunk/`.
Production remains off; no source chunk has been deleted.
