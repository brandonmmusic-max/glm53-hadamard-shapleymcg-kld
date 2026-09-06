# Coupled P8 build receipts

These are preparation-only CPU build records, not GPU correctness or KLD
evidence. V2 failed the resource verifier; V3 corrected the verifier and
passed. Both attempts are retained.

`receipt.json` is byte-identical to the original local receipt.
`build.log` is a readable copy with Docker's two progress-line carriage-return
bytes removed by text transport. `build.log.base64` preserves every original
byte, including those carriage returns. Verify the original stream without
writing another file:

```bash
base64 --decode p8-coupled-image-v2-build/build.log.base64 | sha256sum
base64 --decode p8-coupled-image-v3-build/build.log.base64 | sha256sum
```

Expected original log SHA256:

- V2: `5957534a64d16704c5703cd49590e42142da78736a9a13f073561d13442d9de9`.
- V3: `f97b810767c0e6dcefb15c44f2369dc201ebf6947d2bd147a43c597a84e3dd81`.

Expected receipt SHA256:

- V2: `d599ad0ea1ad2a210cd9f0e2e209709c4301233e54590281d085f33457f36169`.
- V3: `403b073ad557e75da02d5253ff250d6c787424235dd31d6a1ef069509250d493`.
