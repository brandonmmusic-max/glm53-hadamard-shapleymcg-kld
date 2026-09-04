#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 ROLE WORKTREE PROMPT OUTPUT_DIR" >&2
  exit 2
fi

ROLE=$1
WORKTREE=$2
PROMPT=$3
OUTPUT_DIR=$4
CODEX_BIN=${CODEX_BIN:-/home/brandonmusic/.local/bin/codex}

mkdir -p "$OUTPUT_DIR"
{
  printf 'role=%s\n' "$ROLE"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'codex_version=%s\n' "$($CODEX_BIN --version)"
  printf 'worktree=%s\n' "$WORKTREE"
  printf 'head=%s\n' "$(git -C "$WORKTREE" rev-parse HEAD)"
  printf 'prompt_sha256=%s\n' "$(sha256sum "$PROMPT" | awk '{print $1}')"
} >"$OUTPUT_DIR/launch.txt"

set +e
"$CODEX_BIN" exec \
  --ephemeral \
  --json \
  --model gpt-6-astra \
  -c 'model_reasoning_effort="max"' \
  --approve-for-me \
  -C "$WORKTREE" \
  --output-last-message "$OUTPUT_DIR/last-message.txt" \
  - <"$PROMPT" >"$OUTPUT_DIR/events.jsonl" 2>"$OUTPUT_DIR/stderr.log"
status=$?
set -e

{
  printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'exit_status=%d\n' "$status"
  printf 'final_head=%s\n' "$(git -C "$WORKTREE" rev-parse HEAD)"
  printf 'worktree_status_sha256=%s\n' "$(git -C "$WORKTREE" status --porcelain=v1 | sha256sum | awk '{print $1}')"
} >>"$OUTPUT_DIR/launch.txt"
exit "$status"
