#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAMPAIGN=${GLM53_CAMPAIGN_ROOT:-/media/brandonmusic/klcstore/bmxfp4-glm53}
DEST=$ROOT/evidence/opened/codec-v2
RUN_DEST=$ROOT/evidence/historical/kld-v3-runs

mkdir -p "$DEST" "$RUN_DEST"

# Copy only JSON plans, scalar/per-expert analyses, hashes, and run manifests.
# Weight tensors, calibration captures, token arrays, and teacher logits are
# never selected by this snapshotter.
copy_json_tree() {
  local source=$1
  local label=$2
  [ -d "$source" ] || return 0
  while IFS= read -r source_file; do
    local relative=${source_file#"$source"/}
    mkdir -p "$DEST/$label/$(dirname "$relative")"
    cp --reflink=auto "$source_file" "$DEST/$label/$relative"
  done < <(find "$source" -type f -name '*.json' | sort)
}

copy_json_tree "$CAMPAIGN/codec-v2/output-aware/sealed" output-aware-sealed
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/cross-layer" cross-layer
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/composite-3-19-20/sealed-bf16-matched-cf32-v2" matched-composite
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/composite-3-19-20/bf16-layer-attribution-cf32-v1" conditional-fit-layer-attribution
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/supported-subset-3-20/external-v5-v1" external-v5-subset
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/supported-subset-3-20/external-v5-layer-attribution-v1" external-v5-layer-attribution
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/causal-layer-sweep-v1" causal-layer-sweep
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/blockhessian-layer-sweep-v1" blockhessian-layer-sweep
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/layer22-kld/sealed-cf32-v1" layer22-matched-kld
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/layer22-full/receipts" layer22-codec-build-receipts
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/gptq-control-layer22/receipts" layer22-gptq-build-receipts
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/dense-gptq-control-layer22/receipts" layer22-gptq-decode-receipts
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/layer3-full/receipts" layer3-codec-build-receipts
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/layer19-full/receipts" layer19-codec-build-receipts
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/layer20-full/receipts" layer20-codec-build-receipts
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/gptq-control-layer19/receipts" layer19-gptq-build-receipts
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/gptq-control-layer20/receipts" layer20-gptq-build-receipts
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/dense-gptq-control-3-19-20/receipts" layer3-19-20-gptq-decode-receipts
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/layer22-kld/p8-bf16-candidate" layer22-candidate-overlay
copy_json_tree "$CAMPAIGN/codec-v2/output-aware/layer22-kld/gptq-bf16-control" layer22-control-overlay
copy_json_tree "$CAMPAIGN/codec-v2/w6a8-diagnosis/sealed" w6a8-diagnosis
copy_json_tree "$CAMPAIGN/rotation-v5/down-h16-kld-v5-powered/sealed" powered-h16-v5

mkdir -p "$DEST/capture-receipts"
cp --reflink=auto "$CAMPAIGN/evidence/capture-layer-022-prefetch.json" "$DEST/capture-receipts/"

while IFS= read -r run_file; do
  cp --reflink=auto "$run_file" "$RUN_DEST/$(basename "$run_file")"
done < <(find "$CAMPAIGN/kld-v3" -maxdepth 1 -type f \( \
  -name 'run-codec-p8-*.json' -o \
  -name 'run-p8-layer22-*.json' -o \
  -name 'run-v5-powered-downh16-*.json' \
\) | sort)

python3 "$ROOT/scripts/build_public_manifest.py"
python3 "$ROOT/scripts/verify_public_evidence.py"
