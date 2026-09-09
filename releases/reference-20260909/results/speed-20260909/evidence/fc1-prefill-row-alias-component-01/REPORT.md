# Grouped FC1 row-local scratch alias

172 representative exact component comparisons passed on rank0 K4/K5 real checkpoint layers, M4/M16/M4096, including changed-input CUDA graphs. Decode remains original; grouped-prefill changes shared layout only. Reconstruction, FP8 MMA accumulation order, Hadamards, scale boundaries, quantization and barriers remain unchanged.

The CPU ownership checker enumerates all16384 projection halfword stores and every valid-row count0..64. Each row stores gateFP16 and upFP16 together in512 bytes. The warp responsible for that row consumes both projection segments before overwriting only its own512-byte row with128 FP32 values. Existing CTA barriers protect initial publication and final quantizer reads. This check is not a substitute for a GPU race sanitizer.

| Grouped FC1 | Baseline | Row alias |
|---|---:|---:|
| K4 dynamic shared bytes |65536|35840|
| K5 dynamic shared bytes |65536|39936|
| K4 registers/thread |236|236|
| K5 registers/thread |236|238|
| CUDA occupancy API blocks/SM |1|2|

Both default and maximum shared carveout queries report the same1-to2 change on RTX PRO6000 SM120. All compared grouped cubins report stack0/local0. This is theoretical resident capacity from the CUDA API, not measured achieved residency.

At the unchanged grid, paired local-MoE M4096 times differ by+0.22..0.67%, so no speed gain is established. Next candidate launches376 FC1 CTAs for188 SMs while preserving phase0 and FC2 grids. It tests the interaction of reduced shared footprint and sufficient FC1 parallelism. No serving benchmark/adoption yet; production stays stopped.
