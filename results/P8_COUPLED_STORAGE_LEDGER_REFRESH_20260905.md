# Coupled P8 campaign storage-ledger refresh — 2026-09-05

Status: **read-only snapshot and bounded projection pass, with narrow margin.**

Measurement: Unix `1788647073`, `2026-09-05T18:24:33-04:00`.
The original conservative campaign cutoff remains Unix `1788632160`
(`2026-09-05T14:16:00-04:00`); no new baseline was selected.

No file was deleted, encoded, downloaded, built, or moved. No GPU or service
command was run. Production must remain off. This report does not authorize
retirement, launch, KLD capture, image work, or service restoration.

## Verdict

All three layers can be retained simultaneously in both four-chunk and four
TP-sidecar form under the stated 30,000,000,000-byte ceiling, but only narrowly:

```text
5,542,343,936  charged campaign state outside the three-layer root
23,139,643,856 projected three-layer two-copy files and 15 MB receipts/logs
1,030,665,232  reserve: one complete orphan chunk plus 64 MiB short-lag growth
--------------
29,712,653,024 projected aggregate
   287,346,976 remaining below ceiling
```

This is a conservative *defined-scope* pass, not a broad blank check. The
remaining 287 MB is too small for another unbounded image, worktree, capture,
model materialization, or duplicate output. Refresh immediately before layer
22 if any such writer or new campaign branch appears. A stricter flat 1.5 GB
uncertainty reserve would miss the ceiling by 181,987,792 bytes.

At the snapshot, layer 3 had all four chunks and all four sidecars. Layer 20 had
two completed chunks (`0:72`, `72:144`); its remaining chunks had not yet
appeared. The full three-layer root occupied 9,635,642,657 apparent
bytes. Future values below use the largest observed coupled chunk size, so the
active layer-20 chunk and every layer-22 chunk are counted at completion.

## Refreshed charged components

| Component | Apparent bytes | Treatment |
|---|---:|---|
| seven current coupled/audit worktrees, charged in full | 3,121,193,309 | measured |
| all shared Git files modified since cutoff, 1,150 files | 7,598,465 | measured; objects, refs, logs and worktree metadata; no reset baseline |
| coupled-named files outside charged trees | 199 | measured with privileged traversal |
| image preparation plus v2-v9 build directories, nine total | 7,315,738 | measured in full |
| Docker/containerd files modified since cutoff, 998 files | 130,640,876 | measured broad upper bound, includes unrelated growth |
| synthetic fixture and all device/graph receipts below it | 965,085,103 | measured in full |
| maximum historical quality-root simultaneous footprint | 1,310,510,246 | retained prior measured peak, not current 54,658,202 bytes |
| **outside-three-layer subtotal** | **5,542,343,936** | |

The worktree subtotal includes this isolated audit tree in full. The audit file
and its Git commit are later small growth covered by the short-lag reserve.
Docker/containerd breakdown at the snapshot was 42,283,693 bytes of content,
50,343,936 metadata, 36,415,165 snapshots, 1,048,576 Docker image metadata,
25,218 Docker container files, and 524,288 other bytes.

Raw filesystem headroom was 142,148,128,768 bytes on the campaign mount. That
does not alter the user's aggregate-new-data ceiling.

## Layer completion projection

Observed layer-3 files establish the sidecar geometry, and the completed first
two layer-20 chunks establish the conservative chunk maximum:

- exact layer-3 chunks: 3,854,223,760 bytes;
- exact four TP4 sidecars: 3,853,989,824 bytes;
- largest observed 72-expert chunk: 963,556,368 bytes;
- each sidecar: 963,497,456 bytes.

The projection charges 5,000,000 bytes of headers, receipts and logs per layer,
in addition to the actual safetensors sizes:

| Layer | Current completion at snapshot | Completed two-copy projection |
|---:|---|---:|
| 3 | 4 chunks + 4 sidecars complete | 7,713,213,584 |
| 20 | 2 chunks complete; 2 chunks + 4 sidecars projected | 7,713,214,976 |
| 22 | 4 chunks + 4 sidecars projected | 7,713,215,296 |
| **total** | | **23,139,643,856** |

The extra 1,030,665,232-byte reserve is intentionally separate: one entire
maximum-sized chunk covers a residual/orphan final path, and 67,108,864 bytes
covers post-snapshot Git, report, logs and short-lag metadata. The encoder and
packer write directly to final paths, so a failed write must be preserved and
charged rather than silently overwritten.

## Layer-3 retirement contingency

The current defined-scope projection needs **zero** retirement. If the operator
instead requires the stricter 1.5 GB reserve, or post-snapshot growth exceeds
287,346,976 bytes, the minimum capacity action is one layer-3 chunk. The
smallest sufficient exact candidate is:

```text
/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-coupled-three-layer-v1/chunks/layer-003/p8-coupled-layer-003-experts-000-072.safetensors
bytes  963555616
sha256 c062c971e61adb2156b2cd049851d1ff39e79fa0852794133646976dd82cb941
```

Retiring that one file would leave 1,250,902,592 bytes under the present
defined reserve, or 781,567,824 bytes under a flat 1.5 GB reserve.

All four layer-3 chunks are structurally redundant only in the bounded sense
established by the source-exact postwrite receipt; their exact identities are:

| Expert range | Bytes | SHA256 |
|---|---:|---|
| 0:72 | 963,555,616 | `c062c971e61adb2156b2cd049851d1ff39e79fa0852794133646976dd82cb941` |
| 72:144 | 963,555,936 | `38111b6f83a41e306aae9d7097ef38a41fd2afc60071f7c83304954e3f11f196` |
| 144:216 | 963,556,104 | `cf216877d8ad25838072496f8a56d07fee65c0f35bc7d0c952bc800a44ea307d` |
| 216:288 | 963,556,104 | `bc52454f0d73dc65a3607b6098796beaef1453097b2a033f874215c841bb54e7` |

`layer-003-postwrite-abi-v2.json` (SHA256
`b844836319e50f56ad41c9b39a3ea6112d5843b8c94c52249482ec0c2b025540`)
reports `status=pass`, all four source hashes,
all four sidecar hashes and source-exact tensor closure. The separate real
loader result (SHA256
`aa679ae402464fa5adbf241dddd37db1a8ab79b09bb8539b786a3b820aae7048`)
reports `decision=pass` for ranks 0-3 and exact runtime-loaded
tensor hashes. However, both receipts explicitly say
`retirement_authorized=false`; the loader did not execute an MoE MMA. Thus this
audit identifies the minimum technically eligible capacity candidate but does
not authorize its deletion. Preserve all receipts even if the user later
authorizes retirement.

## Reproducible read-only commands

```bash
CUT=1788632160
BASE=/home/brandonmusic/KLC_SANDBOXES
ROOT=/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6

for p in "$BASE"/bmxfp4-glm53-p8-coupled-* \
  "$BASE"/bmxfp4-glm53-p8-storage-ledger-refresh-v1; do
  [ -d "$p" ] && du --apparent-size -sxB1 "$p"
done

find "$BASE/bmxfp4-glm53/.git" -type f \
  -newermt "@$CUT" -printf '%s\n' |
  awk '{s+=$1;n++} END{printf "files=%d bytes=%.0f\n",n,s}'

sudo -n find "$BASE" -xdev \
  \( -path "$BASE/bmxfp4-glm53-p8-coupled-image-v2" \
     -o -path "$BASE/bmxfp4-glm53-p8-coupled-input-order-v1" \
     -o -path "$BASE/bmxfp4-glm53-p8-coupled-integration-v1" \
     -o -path "$BASE/bmxfp4-glm53-p8-coupled-prefill-v1" \
     -o -path "$BASE/bmxfp4-glm53-p8-coupled-scale-kernel-v1" \
     -o -path "$BASE/bmxfp4-glm53-p8-coupled-scale-v1" \
     -o -path "$BASE/bmxfp4-glm53-p8-storage-ledger-refresh-v1" \
     -o -path "$BASE/bmxfp4-glm53/.git" \) -prune -o \
  -type f -newermt "@$CUT" -path '*coupled*' -printf '%s %p\n'

for p in "$ROOT"/p8-coupled-image-preparation-v1 \
  "$ROOT"/p8-coupled-image-v*-build; do
  [ -e "$p" ] && du --apparent-size -sxB1 "$p"
done

du --apparent-size -sxB1 \
  "$ROOT/p8-coupled-fixture-v1" \
  "$ROOT/p8-coupled-three-layer-v1" \
  "$ROOT/tail-v2-p8-exl3-cf32-product-validation-v1b"

find "$ROOT/p8-coupled-three-layer-v1" -type f \
  \( -path '*/chunks/*' -o -path '*/sidecars/*' \) \
  -printf '%s %p\n' | sort -k2

sudo -n find /var/lib/docker /var/lib/containerd -xdev -type f \
  -newermt "@$CUT" -printf '%s\n' |
  awk '{s+=$1;n++} END{printf "files=%d bytes=%.0f\n",n,s}'
```

The 1,310,510,246-byte quality peak is intentionally carried forward from
`results/P8_COUPLED_STORAGE_LEDGER_AUDIT.{md,json}` because the completed raw
capture was retired and the current 54,658,202-byte directory cannot reproduce
that historical simultaneous peak.
