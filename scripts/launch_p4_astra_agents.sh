#!/usr/bin/env bash
set -euo pipefail

REPO=${GLM53_REPO:-/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53}
OUTPUT_ROOT=${GLM53_ASTRA_OUTPUT_ROOT:-/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/agents/p4-astra-v1}
RUNNER=$REPO/scripts/run_p4_astra_agent.sh

launch() {
  local role=$1 unit=$2 worktree=$3 prompt=$4
  if systemctl --user is-active --quiet "$unit.service"; then
    echo "$unit already active" >&2
    return 1
  fi
  systemd-run --user --unit="$unit" --collect \
    --description="GLM-5.3 P4 GPT-6 Astra $role" \
    --property=TimeoutStartSec=infinity \
    "$RUNNER" "$role" "$worktree" "$prompt" "$OUTPUT_ROOT/$role"
}

mkdir -p "$OUTPUT_ROOT"
launch kernel glm53-p4-kernel-astra \
  /home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-astra-p4-kernel \
  "$REPO/agent_prompts/p4_kernel_gpt6_astra.md"
launch codec glm53-p4-codec-astra \
  /home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-astra-p4-codec \
  "$REPO/agent_prompts/p4_codec_gpt6_astra.md"
launch critic glm53-p4-critic-astra \
  /home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-astra-p4-critic \
  "$REPO/agent_prompts/p4_critic_gpt6_astra.md"
