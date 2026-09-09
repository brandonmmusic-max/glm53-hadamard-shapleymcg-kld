# Context shared preference diagnostic

K4/K5 profiles completed successfully; cuCtxSetCacheConfig(PREFER_SHARED) returned success. It did not change actual FC2 function configuration: CachePreferNone,65536B partition,2block shared-memory limit,8theoretical active warps. Durations29.02/33.57us; achieved occupancy14.70/14.84%. This context hint has no demonstrated resource effect and does not warrant serving.

Actual CuTe launch specifies min_blocks_per_mp=2; installed CUTLASS derives a shared-memory function attribute when no explicit carveout is supplied. CUTLASS supports preferred_smem_carveout as a launch argument. Next isolated candidate requests100 there, keeping min_blocks/grid/math unchanged. No claim that the context API return code proves residency.

Probe is single-rank synthetic route real weights, NCU replay overhead. No served result or quality qualification; production remained stopped.
