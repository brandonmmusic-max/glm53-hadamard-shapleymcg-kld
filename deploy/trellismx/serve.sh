#!/usr/bin/env bash
# Current RC5 P8 path. No clocks, power limits, services or benchmark jobs changed.
set -euo pipefail
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1000000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.975}
PORT=${PORT:-8033}
MODEL_NAME=${MODEL_NAME:-glm53-flash-trellismx-p8-k45}
[[ "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]] && (( MAX_MODEL_LEN > 0 && MAX_MODEL_LEN <= 1048576 )) || {
  echo 'MAX_MODEL_LEN must be in 1..1048576; no unvalidated RoPE extension is applied.' >&2; exit 2;
}
for path in /model/config.json /p8-design/design-0.json /p8-design/design-1.json /p8-design/design-2.json /p8-design/transform.json; do
  test -f "$path" || { echo "Required artifact missing: $path" >&2; exit 2; }
done
test -d /p8-sidecars || { echo 'Encoded routed sidecars are required; stock weights alone are not this checkpoint.' >&2; exit 2; }
/opt/venv/bin/python /opt/p8-coupled-runtime/verify_release.py
export GLM53_P8_BAKED_RUNTIME=1 GLM53_P8_VERIFY_MULTIROW_IMPORT=1
export GLM53_P8_NATIVE=1 GLM53_P8_SMALL_M=1 GLM53_P8_FC1_TILE_N=128
export GLM53_P8_FUSED_SCRATCH=1 GLM53_P8_GRID_POLICY=1
export GLM53_P8_FC1_WARP_QUANT=0 GLM53_P8_FC1_BROADCAST_A=1
export GLM53_P8_PREFILL_CHUNK_TOKENS=0 GLM53_P8_ROUTE_CAPTURE=0
export GLM53_P8_DECODE_CAPTURE_V2= VLLM_USE_V2_MODEL_RUNNER=1
export GLM53_P8_NATIVE_LAYERS=3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44
export GLM53_P8_NATIVE_SIDECAR_DIR=/p8-sidecars
export GLM53_P8_NATIVE_DESIGN=/p8-design/design-0.json:/p8-design/design-1.json:/p8-design/design-2.json
export GLM53_P8_NATIVE_TRANSFORM=/p8-design/transform.json
export VLLM_ENABLE_PCIE_ALLREDUCE=1 VLLM_PCIE_ALLREDUCE_BACKEND=b12x
export VLLM_EXL3_PREFILL_TRELLIS=1 VLLM_EXL3_PREFILL_BLOCK_M=128
export VLLM_EXL3_EXT_PATH=/opt/exllamav3
export PYTHONPATH=/opt/p8-mtp-bootstrap:/opt/p8-coupled-runtime:/opt/exllamav3:/opt/infernal-invocation/vllm:/opt/infernal-invocation/b12x
exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /model \
  --served-model-name "$MODEL_NAME" --host 0.0.0.0 --port "$PORT" \
  --language-model-only --tensor-parallel-size 4 --decode-context-parallel-size 1 \
  --dcp-comm-backend a2a --dtype bfloat16 --attention-backend B12X_MLA_SPARSE \
  --kv-cache-dtype nvfp4_ds_mla --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens 8192 --max-num-seqs 32 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-chunked-prefill --enable-prefix-caching \
  --generation-config /model --reasoning-parser glm45 \
  --quantization modelopt --load-format instanttensor "$@"
