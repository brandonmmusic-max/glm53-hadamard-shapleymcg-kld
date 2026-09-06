# V6 diagnostic dispatch failure

2026-09-05. Both the initial diagnostic and the evidence-preserving repeat
failed `selected raw FC1 row 1 retained sentinel words`. The repeat changed
only host evidence persistence, not the image/kernel/fixture/input/gates.

Raw scratch evidence from the repeat:

- 16,896 u32 words; payload writes occur only in rows
  `0,16,32,48,64,80,96,112`, 128 words per written row.
- Raw diagnostic expected four distinct rows per route, 32 written rows total.
- Metadata/count positions instead contain values such as `2105441661` and
  `2105376126`, consistent with packed UE8M0 scale bytes rather than encoded
  expert/route/tile identifiers or write counts.
- 28 words in the supposedly untouched scale tail were modified.
- All 128 input-trace FP32 values are finite, so the input diagnostic did write.

This pattern supports normal quantized FC1 output having been executed instead
of the requested raw-capture subclass. It is not a valid raw-FC1 comparison.
Investigate actual specialization/dispatch and nested compiler caching before
changing numerical math. A class-attribute versus instance-attribute cache-key
collision is a hypothesis only; source verification is still required.

Raw NPZ SHA-256:
`1ceefb924c2ea08fad9c1eadbcfabf4cc5c904d017a8e49d5bd4a8d8939f6d61`.
Local directory:
`/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-fixture-v1/m1-raw-localization-v6-evidence`.
Portable failure and execution receipts are in
`evidence/preparation/p8-raw-localization-v6-evidence/`.

Image `sha256:43b61de24fa8f9323d154705fe37dfe66f39e9954f659895d067b1d99e38e868`;
installed manifest `aa2866e4201dd073a92b2ce172e3be228a6a59a1c30301373ec90ed03b4a00c5`.
Both runs were locked GPU0 synthetic M1 P8, E4M3/UE8M0-K32, K4 4.25-bpw payload
plus coupled metadata; no attention/KV or teacher data exercised. P8 MMA issue
count is twice NVFP4. No closure, KLD or speed result is qualified. Production
remains off. Post-build Docker/containerd cutoff charge: 108,692,202 bytes,
inside the existing storage reserve.
