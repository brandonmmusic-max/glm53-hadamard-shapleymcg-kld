# V3 MXFP6 pilot chunk-count correction

Date: 2026-09-03. The first 6-bpw orchestrator invocation completed and validated all four layer-3 MXFP6 expert-range chunks, then failed closed before candidate construction or inference. The shell array stores each `--chunk` option and its path as separate elements, but the completeness gate incorrectly required four array elements instead of eight.

The gate now requires eight elements, corresponding exactly to four `--chunk PATH` pairs. Quantized tensors, rotation, roles, KLD code, Shapley design, candidate membership, statistical rules, and protected data are unchanged. The failed invocation and all four per-shard logs and receipts remain preserved; execution resumes from those validated chunks at a clean implementation commit.
