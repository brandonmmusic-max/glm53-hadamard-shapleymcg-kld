# P4 independent CPU codec receipt

**Demonstrated:** the versioned alpha-1 MCG/RNE law decodes a cyclic K4
stream into exact native E2M1 nibbles and physical positive E4M3/16 scale
bytes. The scalar oracle agrees at every one of 65,536 MCG states; NVIDIA's
installed host conversion independently agrees at those states and at all
126 positive E4M3 encodings. No state lookup table is used; the largest
numeric lookup array is 1008 bytes.

The final focused run passed **46 tests**. It covers signed zero, midpoint
ties, cyclic states, rectangular tile layouts, scale fitting against
exhaustive SSE, scale addresses and padding, serialization, byte comparisons,
and malformed inputs. `pytest-attempt-1.log` preserves one failed assertion:
the test incorrectly expected alpha-1 MCG to emit the legal ±6 codes. All
state comparisons already passed; the observed alphabet contains 14 codes.
The assertion and documentation were corrected, with the original law kept.
The final review also added explicit string validation for shape metadata
and a scale-fitter invalid-input test. These are developmental corrections,
not a repeated qualification attempt.

For the deterministic **32x128** fixture: 2048 stream bytes + 256 residual
scale bytes + 4 global-scale bytes = **2308 payload bytes**, or
**4.5078125 bpw**. Its 1112 container bytes bring the file to **3420 bytes**,
or **6.6796875 bpw**. Exactly 4.5 bpw describes only stream plus block scales.
The manifest charges every sidecar byte; expectations and receipts are
separate test artifacts.

**Not tested:** GPU execution, MMA register/lane loading, GPU scale repacking,
activation quantization, MoE integration, device determinism, speed, and model
quality/KLD. No protected roles or models were opened; no GPU/service action
was performed. The g++ oracle links only libc, with no CUDA runtime or driver.
The fused implementation and the P8 runtime/schema files are unchanged.

**Compatibility:** this contract explicitly uses alpha 1, native ties-to-even,
and signed zero. The legacy P4 LUT law and the alpha-2 P8 law must not be
relabeled as this format. A GPU probe must match the declared law and these
expected bytes before making a device-closure claim. A general model-weight
trellis encoder and a quality or speed advantage are not established.

See [the format contract](../../../../docs/P4_CODEC.md), `receipt.json` for
commands, identities, scope and artifact hashes, and `fixture.expected.json`
for GPU-probe inputs. From the repository root:

```bash
sha256sum -c evidence/opened/codec-v2/p4-astra/MANIFEST.sha256
CUDA_VISIBLE_DEVICES='' python3 -m glm53_nvfp4.p4_fixture evidence/opened/codec-v2/p4-astra
```

ExLlamaV3/turboderp supplies the procedural constants and stream/lane layout.
KQuant/QSRT and the vendored `w4a8_trellis` port retain their attribution and
unverified snapshot license status in `THIRD_PARTY_NOTICES.md`.
