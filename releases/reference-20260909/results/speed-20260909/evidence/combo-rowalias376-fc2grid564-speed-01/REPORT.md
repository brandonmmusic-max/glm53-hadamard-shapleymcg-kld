# Combined rowalias376 and FC2grid564 serving screen

Completed. Experimental candidate retained, not adopted; no clear overall served gain or fuller-run justification from this single screen. Production and candidate both stopped after successful unit completion. Original production image/config/checkpoint unchanged.

|Measurement|Prior rowalias376+overlap4096|Combination|
|---|---:|---:|
|32K cold prefill t/s|8218|8196.0|
|64K cold prefill t/s|Not measured|8130.0|
|8K C1 speculative decode t/s|176.64513|181.96215|
|8K C4 aggregate speculative decode t/s|320.29388|312.22045|

C4 aggregate divided by4:78.055t/s. Prefill TTFT32K3.998s and64K8.062s,5and3cold requests respectively. Prior separate FC2-only64K run was7866t/s,8.332sTTFT; different image/server, so observed+3.36%is exploratory comparison, not matched causal gain.

One server, one pass,20-second windows, decode cap8192. No request errors, underfill, warmup timeout or capacity limitation. Current MTP acceptance fields C10.513889, C40.524194; prior0.412037/0.534946. MTP variation can influence speculative output rates. C1 rises while C4 falls relative to prior; no reliable decode win established. No replication/confidence interval; requests within a server are subsamples.

Runtime verifiedTP4/DCP4,MTP3probabilistic,NVFP4KV,maxseq24,batch4096,FULL_AND_PIECEWISE on4ranks,4x300W. Shared-expert threshold4096 independently read from installed envs. Startup KV23,418,300tokens, distinct from observed benchmark capacity. Image `sha256:3016bb756e35b0aa2c7b481ef5c8dd3455ac23b9029ed7d179c22594bc03d147`. Source hashes matched the tested combined FC1/FC2 modules. Benchmark SHA `7239958032ec10781d7db3efca23a7602bd9aeb94455a723e2634ab7d6fe546a`.

Combination: groupedFC1 row-local shared alias/grid376; shared-expert aux-stream eligibility through4096; directFC2 preferred_smem_carveout100/grid564. Native FP8MMA/checkpoint unchanged.172exact combined component comparisons passed against original; full-model quality/KLD remains unqualified. Raw smoke retained; it is not a quality qualification. No new baseline throughput run, power change, P4 work or publication.

Evidence: plan-01.json and its hash, SUMMARY.json, image source, results-01/combo-rowalias376-fc2grid564/rep-1/prefill.json and decode-cap8192.json, runtime/source verification, metric snapshots and telemetry. Component evidence ../combo-rowalias376-fc2grid564-component-01/REPORT.md; FC2-only64K ../fc2-grid564-prefill64k-01/REPORT.md. Targets300C1/20000prefill remain unmet.
