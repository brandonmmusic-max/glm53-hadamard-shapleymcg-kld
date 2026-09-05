# V7 execution decisions

1. Raw localization and conditional M1 closure were authorized before execution
   by `experiments/p8-coupled-v7-execution.json`. Their completed results are
   recorded in `P8_COUPLED_V7_DEVICE_RESULT.md`; both required stages passed.
2. Before opening prefill results, execute the existing frozen M2/M64/M65
   protocol in `scripts/run_p8_coupled_prefill_device_closure.py` on the same
   v7 immutable image and fixture, GPU0 under the model-stack lock. Preserve
   the existing exact carrier, five-repeat determinism and numerical gates.
   No retuning or threshold change. Reserve another 67,108,864 evidence bytes;
   aggregate conservative campaign bound becomes 8,342,606,336 bytes, below
   the user's 30,000,000,000-byte limit. No new fixture or model allocation.
   Stop dependent model execution on any failure. Production remains off.
