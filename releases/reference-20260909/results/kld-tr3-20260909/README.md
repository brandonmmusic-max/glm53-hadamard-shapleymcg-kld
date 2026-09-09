# Matched-data TR3 4bpw and TrellisMX reference KLD

| MLA KV cache | TrellisMX reference | TR3 / EXL3 4bpw | TR3 minus reference |
| --- | ---: | ---: | ---: |
| fp8 | 0.0319451732 | 0.0281899278 | -0.0037552454 |
| nvfp4 | 0.0354562238 | 0.0304785381 | -0.0049776857 |

Lower KL(teacher || student) is better on these measured inputs. The32windows,
teacher hashes, exact token histories,2046true-decode rows/window and FP64 scorer
match the previous reference measurement. Equal mean of32windowmeans; BCa95,
20000resamples, seed20260902. Per-window values and paired intervals in comparison.json.
FP8 then NVFP4, one server preparation per arm; all prefix-hit counters zero.

TR3 means the owner's uniform4bpw GLM5.3 Flash checkpoint. Its TR3 and EXL3 Hub names
resolve to the same revision/manifest, not two different comparison models.
No stock NVFP4-weight model was substituted; FP8/NVFP4 here label KV caches.

This is a matched-data system comparison, not an isolated codec ablation:
TR3 uses its compatible Jovian r10 TP4/EP4/DCP4 runtime and source-native nonrouted
weights; TrellisMX uses r27 TP4/EPoff/DCP4 and its carrier. Cache block layout and
kernel implementations differ. Both use MTPoff,maxseq1,batch4096,maxlen1M,GMU.97,
4x300W. TR3 loader requiresEP4 for this checkpoint atTP4; checkpoint unmodified.
Capture seam CPU tests preserve pre-mask logits and forced histories; immutable
image/source and fresh full checkpoint verification hashes accompany the results.

Already-opened conditional-fit development data, not untouched-final qualification
or independent replication. Window intervals do not quantify server-run variability,
and these short windows do not establish long-context or general answer quality.
Raw logits were retired after verified hashes and durable per-row scores/receipts.
Retained numerical scores, geometry, metadata, startup contract and zero-prefix-hit
receipts were audited; no independent recapture is claimed.
