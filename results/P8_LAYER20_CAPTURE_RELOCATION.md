# Layer20 calibration relocation amendment

The first layer20 launch failed before encoding: the old klcstore layer20
directory retained only a partial-capture receipt, not its hidden-state file.
Failure logs and execution receipt remain under the original layer20-000-072
names. No codec output or chunk receipt was created.

Verified alternate root:
`/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/fit-capture-l20-l22-v1`.
Its capture-manifest SHA256 is identical to the pinned V3 source:
`f1a6fe7b8828b3461e81ee533d417dd1524355ef6a205145850116560197f81a`.

Before retry, CPU verification compared all64 current roles-v5 fit IDs and
token hashes with the manifest, required their exact window-index set to match
the partial-fit receipt, and reread all162 recorded hidden/route-id/route-weight
ranges. All1,080,033,280 bytes matched their recorded SHA256 values. Only fit
ranges were read; no protected logits or roles were opened. The partial receipt
carries an older overall roles hash, so matching current per-window IDs/tokens
and index membership was necessary rather than assuming the roles were equal.

Amendment changes only `--capture-root` and uses fresh `capture-relocation-v2`
launch/log filenames. Encoder/design/fit sampling/transform/bits remain pinned.
The output codec path is unchanged and was absent; existing attempts are not
overwritten. No data copied, downloaded, removed or newly allocated.
Production remains off. This is a calibration-availability repair, not a KLD
result or a candidate change.
