# GLM-5.3-Flash BF16 to NVFP4 V2 report

Status: **STOPPED** at selection evidence level. No publication or Hub upload was authorized or performed.

## Primary result

- Candidate mean KLD: 0.04771367 nats.
- Stock NVFP4 mean KLD: 0.04542187 nats.
- Relative improvement: -5.05%.
- Equal-window paired delta: 0.00229180; BCa 95% CI [-0.00161453, 0.00682856].
- Registered rule at this terminal stage: candidate selection mean must be lower than stock before candidate freeze and confirmation.

## Quantization gates

All 42 routed-expert layers and all 288 experts per layer passed the matched RTN weighted-error gate. Ratios below 1 favor GPTQ.

- gate: mean layer ratio 0.6666; worst expert ratio 0.9321.
- up: mean layer ratio 0.6676; worst expert ratio 0.9325.
- down: mean layer ratio 0.7207; worst expert ratio 0.9772.

The candidate changes exactly 108864 routed-expert tensors across 168 chunks. Non-expert tensors remain supplied by the stock carrier.

## Role separation

- Selection comparison: candidate 0.04771367, stock 0.04542187, decision `stop`.
- Confirmation: not opened; the preregistered selection stop preserved all 32 confirmation windows.
- Legacy final windows: not used as new final evidence because they had already been opened.

## Runtime and limitations

The servability gates proved the ModelOpt checkpoint, FLASHINFER_CUTLASS NVFP4 MoE backend, and FLASHINFER_MLA_SPARSE_SM120 attention in TP4/EP4/DCP4 eager mode. Calibration used immutable BF16 activation/router captures and group-16 block Hessians; it did not rerun the 642 GB BF16 checkpoint locally or propagate quantized activations causally across layers. This is one quantization campaign, so it does not estimate between-campaign variance. A selection stop means performance benchmarks are skipped and no confirmation-level claim is available.

Benchmark summary receipts found: 0. Performance, Estonia, and LAVD are separate post-quality regimes and do not alter the KLD decision.

## Principal artifacts

- Candidate: `/media/brandonmusic/klcstore/bmxfp4-glm53/candidates/uniform-gptq`
- Terminal receipt: `/media/brandonmusic/klcstore/bmxfp4-glm53/evidence/selection-terminal.json`
- Decision analysis: `/media/brandonmusic/klcstore/bmxfp4-glm53/evidence/selection-analysis.json`
- Experiment record: `/media/brandonmusic/klcstore/bmxfp4-glm53/experiment-record.json`
