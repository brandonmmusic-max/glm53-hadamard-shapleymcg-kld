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
