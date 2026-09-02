#!/usr/bin/env bash
# BMXFP4 pilot orchestration on 4 GPUs.  Idempotent-ish: skips stages whose receipts exist.
set -uo pipefail
CODE=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-qwen3-30b-a3b/bmxfp4
PY=/home/brandonmusic/klc-env/bin/python
ROOT=/media/brandonmusic/klcstore/bmxfp4
LOG=$ROOT/logs/campaign.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ local gpu=$1; shift; local name=$1; shift; log "START gpu$gpu $name"; CUDA_VISIBLE_DEVICES=$gpu "$PY" "$@" > "$ROOT/logs/$name.out" 2>&1; local rc=$?; log "END gpu$gpu $name rc=$rc"; return $rc; }

stage=${1:-all}

if [[ $stage == all || $stage == stage0 ]]; then
  # ---- Stage 0: teacher (GPU0) in parallel with Hessians for layers 0-35 (GPUs 1-3, 12 layers each)
  [[ -f $ROOT/teacher/b0-receipt.json ]] || run 0 teacher "$CODE/run_teacher.py" --device cuda:0 --panels selection,final,confirmation,wikitext --b0-runs 3 &
  [[ -f $ROOT/hessians/receipt-0-11.json ]]  || run 1 hess-0-11  "$CODE/run_hessians.py" --device cuda:0 --layers 0-11 &
  [[ -f $ROOT/hessians/receipt-12-23.json ]] || run 2 hess-12-23 "$CODE/run_hessians.py" --device cuda:0 --layers 12-23 &
  [[ -f $ROOT/hessians/receipt-24-35.json ]] || run 3 hess-24-35 "$CODE/run_hessians.py" --device cuda:0 --layers 24-35 &
  wait
  [[ -f $ROOT/hessians/receipt-36-47.json ]] || run 0 hess-36-47 "$CODE/run_hessians.py" --device cuda:0 --layers 36-47
fi

if [[ $stage == all || $stage == stage1 ]]; then
  # ---- Stage 1: control + rotation arms on selection (W4A16 + W4A4), 4 in parallel
  run 0 B3   "$CODE/run_arm.py" --arm B3  --device cuda:0 --method rtn  --panels selection --a4 &
  run 1 B3p  "$CODE/run_arm.py" --arm B3p --device cuda:0 --method gptq --panels selection --a4 --save-weights &
  run 2 B5   "$CODE/run_arm.py" --arm B5  --device cuda:0 --method gptq --rotation had16 --panels selection --a4 &
  run 3 B4s1 "$CODE/run_arm.py" --arm B4s1 --device cuda:0 --method gptq --rotation random:1 --panels selection &
  wait
  run 0 B4s2 "$CODE/run_arm.py" --arm B4s2 --device cuda:0 --method gptq --rotation random:2 --panels selection &
  run 1 B4s3 "$CODE/run_arm.py" --arm B4s3 --device cuda:0 --method gptq --rotation random:3 --panels selection &
  run 2 B4s4 "$CODE/run_arm.py" --arm B4s4 --device cuda:0 --method gptq --rotation random:4 --panels selection &
  run 3 B4s5 "$CODE/run_arm.py" --arm B4s5 --device cuda:0 --method gptq --rotation random:5 --panels selection &
  wait
fi

if [[ $stage == all || $stage == stage2 ]]; then
  # ---- Stage 2: candidates for the ladder (identity rotation unless the rotation verdict says otherwise)
  ROT=${ROTATION:-identity}
  run 0 cand "$CODE/run_arm.py" --arm cand-$ROT --device cuda:0 --mode candidates --method gptq --rotation $ROT --layers 0-11 &
  run 1 cand1 "$CODE/run_arm.py" --arm cand-$ROT --device cuda:0 --mode candidates --method gptq --rotation $ROT --layers 12-23 &
  run 2 cand2 "$CODE/run_arm.py" --arm cand-$ROT --device cuda:0 --mode candidates --method gptq --rotation $ROT --layers 24-35 &
  run 3 cand3 "$CODE/run_arm.py" --arm cand-$ROT --device cuda:0 --mode candidates --method gptq --rotation $ROT --layers 36-47 &
  wait
fi
log "stage $stage finished"
