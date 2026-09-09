# Shared-expert dependency audit

Production was verified stopped; source copied without starting it.

Installed envs.py defines the shared-expert stream token threshold as256 and production has no override. Its disable flag is0. SharedExperts._determine_shared_experts_order excludes inputs larger than this threshold. The installed GLM shared MLP has no shard_sequence_parallel attribute; shared down projection uses reduce_results=False. Standard runner combines shared and routed outputs before the required final reduction when routed output is TP-local. No duplicate reduction removal is justified.

The initial auxiliary-stream wait precedes routed dispatch in _forward_impl. The shared path queues asynchronously on its stream and joins before consuming shared output. Python enqueue order alone does not establish GPU serialization. Keep both waits.

Existing historical Nsight exports lack graph edges; CUDA event timestamps are all zero. They cannot establish a complete critical path or prove waits removable. Production maxseq24 also differs from historical traced maxseq16.

Next exploratory candidate raises the existing threshold to4096 on the retained row-alias376 image. No arithmetic/kernel source changes; no barriers removed. This extends eligibility, but actual achieved overlap is not proven by configuration. Competition for SMs, registers, bandwidth and power can erase or reverse benefit. Screen only32K cold prefill and8K C1/C4 decode, one pass; preserve original production stopped. Exact plan and source hashes are in ../shared-prefill-overlap4096-speed-quick01/plan-01.json.

## Collective follow-up

Installed B12X adapter and production startup agree on one-shot86016B, two-shot786432B, DMA disabled. Launcher libexec/serve-glm53-flash-nvfp4-dflash2.sh lines88-100 explicitly selects DMA off for DCP>1 because prior measured full-CKV prefill was slower with DMA. This is documented historical rationale, not newly replicated evidence. Do not propose simply enabling DMA or extending the two-shot ceiling as an established improvement. Adapter source documents two-shot matching NCCL at1MiB and above.

Next diagnostic is focused FC2 decode K4/K5 PC sampling; plan in ../fc2-decode-pcsamp-01. Historical C1cap8192 trace2 device0 FC2 summed durations67.179ms K4 and51.993ms K5 over2700/1836 launches. These sums do not define wall-time or a speedup bound.
