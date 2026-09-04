#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAMPAIGN=${GLM53_CAMPAIGN_ROOT:-/media/brandonmusic/klcstore/bmxfp4-glm53}
SOURCE=$CAMPAIGN/exact-v10/selective-h16
DEST=$ROOT/evidence/opened/exact-v10-selective-h16
RUN_DEST=$ROOT/evidence/historical/kld-v3-runs

mkdir -p "$DEST" "$RUN_DEST"

# These files contain only opened conditional-fit/selection metrics, build
# identities, and pre-score freezes. They do not contain model weights,
# teacher logits, token arrays, or unopened confirmation inputs.
while IFS= read -r source_file; do
  relative=${source_file#"$SOURCE"/}
  mkdir -p "$DEST/$(dirname "$relative")"
  cp --reflink=auto "$source_file" "$DEST/$relative"
done < <(find "$SOURCE" -type f \( \
  -name 'build.json' -o \
  -name 'conditional-fit-vs-stock.json' -o \
  -name 'selection-wave*-freeze.json' -o \
  -name 'selection-wave*-vs-stock.json' \
\) | sort)

while IFS= read -r run_file; do
  cp --reflink=auto "$run_file" "$RUN_DEST/$(basename "$run_file")"
done < <(find "$CAMPAIGN/kld-v3" -maxdepth 1 -type f \
  -name 'run-exact-v10-h16-layer-*.json' -o \
  -name 'run-exact-v10-h16-layers-*.json' -o \
  -name 'run-exact-v10-wave2-*.json' -o \
  -name 'run-exact-v10-wave3-*.json' | sort)

python3 "$ROOT/scripts/build_public_manifest.py"
python3 "$ROOT/scripts/verify_public_evidence.py"
