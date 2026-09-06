# Layer22 stage decisions

1. Layer20 encoding finished exit0 for all288 experts, CPU TP4 packing and
   postwrite source-exact checks passed, and the real four-rank loader passed
   with result SHA256
   `371c1b88d358ceb40ad14db8dd22aeb96a3b016819c444d208317ebe7e9d9929`.
   Loader evidence is not MMA execution or KLD.
2. Before layer22 encoding, pin `p8-coupled-layer22-encoding-v1.json` with the
   unchanged V3 encoder, graph result, layer20 loader result and fresh layer22
   fit-range verification. Execute fixed ranges0:72,72:144,144:216,216:288.
   No new transform draws or parameter tuning. Same256 domain-balanced fit
   samples, native K4 MCG E4M3/UE8M0 target, coupled H512/H128/suh/svh.
3. The encoding-stage conservative total is26,389,877,768 bytes: observed
   three-layer-root15,416,868,600 +4.4GB new chunks/logs +5,542,343,936 external
   state +1,030,665,232 orphan/lag reserve. Layer22 packing remains a separate
   gate. No new image, worktree, capture, source retirement or production
   restoration. Check backend USER scope and timer SYSTEM scope before each
   chunk, require idle GPU0 and bounded cooldown under the exclusive lock.

Conditions: attention/KV N/A during encoding; target MoE native coupled P8,
target activations E4M3, approximately4.25398bpw including metadata. P8 has
twice the NVFP4 MMA issue count. No new end-to-end KLD result is claimed.

4. Before packing results, pin `p8-coupled-layer22-pack-v1.json` and its
   CPU-only runner: four valid chunks required, unchanged packer/verifier
   source hashes, no overwrite, failed pack prevents verifier. Five runner
   tests pass. The complete artifact-directory bound is23.3GB; refreshed
   external state plus orphan/lag reserve is6,573,009,168 bytes, yielding a
   conservative29,873,009,168-byte bound. No new image/worktree/capture allowed
   from this packing plan; refresh if unrelated campaign growth exceeds it.
5. CPU control-input preflight: `validate_identity_sidecars` successfully
   rehashed all12 stored identity-P8 sidecars on layers3/20/22, totaling
   11,551,121,952 bytes, against the existing all42 manifest and design.
   The stock carrier is `/home/brandonmusic/models/GLM-5.3-Flash-NVFP4`, NOT
   the old recipe's pseudoquant model mount. Its config SHA256 is
   `676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996`,
   index SHA256 `0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb`.
   These metadata checks are not a fresh hash audit of all stock weight shards.
6. Pin `p8-layer22-loader-execution-v1.json` before its result: same immutable
   v9 image and loader harness, explicit layer22 protocol, all four ranks,
   source-exact postwrite prerequisite and fresh storage/production-off gate.
7. Read-only CF32 availability check via `load_role_inputs(...,
   verify_teacher_bytes=False)` passed all32 token-file hashes, manifest/Hub
   identities, teacher file sizes and stored F32 tensor/header checks, with
   exactly8 windows in each domain. Full teacher byte rehash remains required
   at runtime-manifest preparation; this check must not be represented as it.
   No protected logits were opened.
