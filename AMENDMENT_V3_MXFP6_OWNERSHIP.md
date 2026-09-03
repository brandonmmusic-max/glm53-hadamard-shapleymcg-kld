# V3 MXFP6 bind-mount ownership correction

Date: 2026-09-03. After the four layer-3 MXFP6 shards passed their exact-payload validation, the host-side mixed-candidate builder failed before completing its receipt and before any runtime inference. The pinned Docker image writes bind-mounted outputs as root with mode `0600`; the unprivileged evidence builder therefore could not open the shard even though it existed.

The producer now applies `0644` to only the completed MXFP6 shard, JSON receipt, and log files for the layer it just produced. The already-produced layer-3 artifacts receive the same permission-only correction. File contents, hashes, quantization, candidate membership, role data, runtime arithmetic, and decision rules do not change. The failed builder attempt remains visible, and execution resumes at a clean implementation commit before the runtime canary or any Shapley target score.
