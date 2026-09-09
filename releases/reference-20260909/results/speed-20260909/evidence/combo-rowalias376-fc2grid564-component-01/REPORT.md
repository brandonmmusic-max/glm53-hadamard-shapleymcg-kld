# Combined FC1/FC2 component validation

172exact comparisons passed, rank0K4/K5M4/M16/M4096. Eager stress covers signs, concentrated routes, zeros and nonfinites; graphs replay with changed finite inputs. Compared complete TP-local outputs, route outputs and intermediate regression. No full-model KLD or shared-expert scheduling qualification.

|Rate|M|Capture|Paired graph-time change|
|---|---:|---:|---:|
|K4|4|0|-0.41%|
|K4|4|1|-0.60%|
|K4|16|0|-6.14%|
|K4|16|1|-4.89%|
|K4|4096|0|-15.57%|
|K4|4096|1|-15.83%|
|K5|4|0|-2.86%|
|K5|4|1|-3.38%|
|K5|16|0|-5.15%|
|K5|16|1|-4.26%|
|K5|4096|0|-15.93%|
|K5|4096|1|-15.65%|

Negative is faster. Five balanced samples of20replays per capture, fixed synthetic routes and real representative weights. This composition retains grouped-prefill component benefit and direct-decode candidate behavior; it does not establish served throughput. Shared-expert threshold4096 is applied only in the subsequent serving image launch.

Original production remained stopped. Sources and inputs are pinned in plan and raw receipts. Subsequent user-requested serving screen is32K/64Kprefill and8KC1/C4 only.
