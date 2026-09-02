#!/usr/bin/env bash
# Phases C and D with the selection-role rotation decision applied (ROT env, default had16).
set -uo pipefail
CODE=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-qwen3-30b-a3b/bmxfp4
PY=/home/brandonmusic/klc-env/bin/python
ROOT=/media/brandonmusic/klcstore/bmxfp4
LOG=$ROOT/logs/campaign.log
ROT=${ROT:-had16}
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ local gpu=$1; shift; local name=$1; shift; log "START gpu$gpu $name"; CUDA_VISIBLE_DEVICES=$gpu "$PY" "$@" > "$ROOT/logs/$name.out" 2>&1; local rc=$?; log "END gpu$gpu $name rc=$rc"; return $rc; }

phase=${1:?phase}
case $phase in
C)
  log "phase C (rotation for candidates: $ROT)"
  run 0 cand-a "$CODE/run_arm.py" --arm cand-$ROT --device cuda:0 --mode candidates --method gptq --rotation $ROT --layers 0-23 &
  run 1 cand-b "$CODE/run_arm.py" --arm cand-$ROT --device cuda:0 --mode candidates --method gptq --rotation $ROT --layers 24-47 &
  run 2 learnR-I "$CODE/learn_rotation.py" --device cuda:0 --init identity --out $ROOT/arms/learnedR-identity --steps ${LEARN_STEPS:-80} --layers 0-47 &
  run 3 learnR-H "$CODE/learn_rotation.py" --device cuda:0 --init had16 --out $ROOT/arms/learnedR-had16 --steps ${LEARN_STEPS:-80} --layers 0-47 &
  wait
  run 0 E1a "$CODE/run_exl3_anchor.py" --arm E1a --recon $ROOT/exl3-recon/4.0bpw --scope full --device cuda:0 --panels selection,final &
  run 1 B6i "$CODE/run_arm.py" --arm B6i --device cuda:0 --method gptq --rotation learned:$ROOT/arms/learnedR-identity --panels selection &
  run 2 B6h "$CODE/run_arm.py" --arm B6h --device cuda:0 --method gptq --rotation learned:$ROOT/arms/learnedR-had16 --panels selection &
  run 3 B5p "$CODE/run_arm.py" --arm B5p --device cuda:0 --method gptq --rotation had16 --permutation diag_band --panels selection &
  wait
  log "phase C done"
  ;;
D)
  B5=$($PY -c "import json;print(json.load(open('$ROOT/seals/bytes-exl3-5.0bpw.json'))['routed_expert_payload_bytes'])")
  B4=$($PY -c "import json;print(json.load(open('$ROOT/seals/bytes-exl3-4.0bpw.json'))['routed_expert_payload_bytes'])")
  log "phase D budgets: E1b routed bytes=$B5  E1a routed bytes=$B4 (candidates cand-$ROT)"
  $PY $CODE/allocate.py --candidates $ROOT/arms/cand-$ROT/candidates --out $ROOT/arms/alloc-E1b-$ROT --budget-bytes $B5 --name E1b > $ROOT/logs/alloc-E1b.out 2>&1
  $PY $CODE/allocate.py --candidates $ROOT/arms/cand-$ROT/candidates --out $ROOT/arms/alloc-E1a-$ROT --budget-bytes $B4 --name E1a > $ROOT/logs/alloc-E1a.out 2>&1
  log "alloc E1b: $(tr '\n' ' ' < $ROOT/logs/alloc-E1b.out | head -c 400)"
  log "alloc E1a: $(tr '\n' ' ' < $ROOT/logs/alloc-E1a.out | head -c 400)"
  run 0 B8  "$CODE/run_arm.py" --arm B8  --device cuda:0 --method gptq --rotation $ROT --tiermap $ROOT/arms/alloc-E1b-$ROT/tiermap.json --panels selection,final,wikitext --a4 --save-weights &
  run 1 B8a "$CODE/run_arm.py" --arm B8a --device cuda:0 --method gptq --rotation $ROT --tiermap $ROOT/arms/alloc-E1a-$ROT/tiermap.json --panels selection,final --a4 &
  run 2 B5-final "$CODE/run_arm.py" --arm B5-final --device cuda:0 --method gptq --rotation had16 --panels final,wikitext --a4 &
  run 3 E1a-experts "$CODE/run_exl3_anchor.py" --arm E1a-experts --recon $ROOT/exl3-recon/4.0bpw --scope experts --device cuda:0 --panels selection,final &
  wait
  run 0 B3p-final "$CODE/run_arm.py" --arm B3p-final --device cuda:0 --method gptq --panels final,wikitext --a4 &
  wait
  $PY $CODE/report.py --out $ROOT/REPORT.md > $ROOT/logs/report.out 2>&1
  log "phase D done"
  ;;
esac
