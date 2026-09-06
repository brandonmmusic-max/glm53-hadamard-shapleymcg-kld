# Activation quantizer dependency pin

Before any coupled candidate encoding, preparation V3 adds the SHA-256 of
canary_mxfp6_reap.py, which supplies the encoder's imported _qdq_e4m3_k32.
V2 pinned the caller but omitted this numerical dependency. The encoder now
checks this hash before CUDA allocation. V1/V2 receipts remain preserved.

V3 preparation SHA-256:
4ebb96dd9d555fc18f24fb5f4d380216e1de30327a68d7aae89fc10f41878695

The transform and scale bytes did not change. The pilot proposal now names
V3; it remains a proposal pending device closure and execution sealing.

Validation: real V3 design and exact Flash scale-source validation; injected
activation-quantizer hash drift rejected before CUDA; 16 focused encoder and
sidecar tests passed. CPU quantizer reconstruction matched runtime reference
exactly on seeded 8x4096 fixtures at multipliers 0, 2^-16, 1 and 2^16. These
fixtures do not establish device payload/scale-byte closure or KLD improvement.
