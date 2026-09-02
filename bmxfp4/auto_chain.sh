#!/usr/bin/env bash
# Phased autopilot for the BMXFP4 pilot.  Usage: auto_chain.sh A|B|C|D
set -uo pipefail
CODE=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-qwen3-30b-a3b/bmxfp4
PY=/home/brandonmusic/klc-env/bin/python
ROOT=/media/brandonmusic/klcstore/bmxfp4
IMG=verdictai/glm53-exl3-k3:jovian-r10-tp4-dcp4-r7-fused-v1
LOG=$ROOT/logs/campaign.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ local gpu=$1; shift; local name=$1; shift; log "START gpu$gpu $name"; CUDA_VISIBLE_DEVICES=$gpu "$PY" "$@" > "$ROOT/logs/$name.out" 2>&1; local rc=$?; log "END gpu$gpu $name rc=$rc"; return $rc; }
wait_pid(){ while kill -0 "$1" 2>/dev/null; do sleep 20; done; }
recon(){ # $1 gpu, $2 bpw
  log "START gpu$1 exl3-recon-$2"
  docker run --rm --gpus "\"device=$1\"" -v $ROOT:/data -v $CODE:/code:ro --entrypoint python3 $IMG /code/exl3_reconstruct.py --model /data/models/turboderp-exl3-$2bpw --out /data/exl3-recon/$2bpw > $ROOT/logs/exl3-recon-$2.out 2>&1
  log "END gpu$1 exl3-recon-$2 rc=$?"
}

phase=${1:?phase}
case $phase in
A)
  log "phase A: waiting for downloads (qwen pid ${QWEN_PID:-none}, exl3-5 pid ${EXL5_PID:-none}, exl3-4 pid ${EXL4_PID:-none})"
  [[ -n ${EXL5_PID:-} ]] && wait_pid $EXL5_PID
  n5=$(ls $ROOT/models/turboderp-exl3-5.0bpw/*.safetensors 2>/dev/null | wc -l); log "exl3-5.0 shards: $n5"
  $PY $CODE/bytes_exl3.py $ROOT/models/turboderp-exl3-5.0bpw > $ROOT/seals/bytes-exl3-5.0bpw.json 2>&1; log "bytes 5.0: $(head -c 200 $ROOT/seals/bytes-exl3-5.0bpw.json | tr '\n' ' ')"
  recon 0 5.0 &
  RECON5=$!
  [[ -n ${QWEN_PID:-} ]] && wait_pid $QWEN_PID
  nq=$(ls $ROOT/models/Qwen3-30B-A3B/*.safetensors 2>/dev/null | wc -l); log "qwen shards: $nq (expect 16)"
  if [[ $nq -lt 16 ]]; then log "ABORT: qwen download incomplete"; exit 2; fi
  run 0 teacher "$CODE/run_teacher.py" --device cuda:0 --panels selection,final,confirmation,wikitext --b0-runs 3 &
  run 1 hess-0-11  "$CODE/run_hessians.py" --device cuda:0 --layers 0-11 &
  run 2 hess-12-23 "$CODE/run_hessians.py" --device cuda:0 --layers 12-23 &
  run 3 hess-24-35 "$CODE/run_hessians.py" --device cuda:0 --layers 24-35 &
  wait
  run 1 hess-36-47 "$CODE/run_hessians.py" --device cuda:0 --layers 36-47 &
  H47=$!
  [[ -n ${EXL4_PID:-} ]] && wait_pid $EXL4_PID
  $PY $CODE/bytes_exl3.py $ROOT/models/turboderp-exl3-4.0bpw > $ROOT/seals/bytes-exl3-4.0bpw.json 2>&1; log "bytes 4.0: $(head -c 200 $ROOT/seals/bytes-exl3-4.0bpw.json | tr '\n' ' ')"
  wait $RECON5
  recon 0 4.0 &
  wait $H47
  wait
  log "phase A done"
  ;;
B)
  # control + rotation arms on selection; EXL3 anchors need the recon dirs
  run 0 B3   "$CODE/run_arm.py" --arm B3  --device cuda:0 --method rtn  --panels selection --a4 &
  run 1 B3p  "$CODE/run_arm.py" --arm B3p --device cuda:0 --method gptq --panels selection,final,wikitext --a4 --save-weights &
  run 2 B5   "$CODE/run_arm.py" --arm B5  --device cuda:0 --method gptq --rotation had16 --panels selection --a4 &
  run 3 B4s1 "$CODE/run_arm.py" --arm B4s1 --device cuda:0 --method gptq --rotation random:1 --panels selection &
  wait
  run 0 E1b "$CODE/run_exl3_anchor.py" --arm E1b --recon $ROOT/exl3-recon/5.0bpw --scope full --device cuda:0 --panels selection,final,wikitext &
  run 1 E1b-experts "$CODE/run_exl3_anchor.py" --arm E1b-experts --recon $ROOT/exl3-recon/5.0bpw --scope experts --device cuda:0 --panels selection,final,wikitext &
  run 2 B4s2 "$CODE/run_arm.py" --arm B4s2 --device cuda:0 --method gptq --rotation random:2 --panels selection &
  run 3 B4s3 "$CODE/run_arm.py" --arm B4s3 --device cuda:0 --method gptq --rotation random:3 --panels selection &
  wait
  log "phase B done"
  ;;
C)
  # candidates (identity) on 2 GPUs, learned rotations on 2 GPUs, then E1a anchors
  run 0 cand-a "$CODE/run_arm.py" --arm cand-identity --device cuda:0 --mode candidates --method gptq --layers 0-23 &
  run 1 cand-b "$CODE/run_arm.py" --arm cand-identity --device cuda:0 --mode candidates --method gptq --layers 24-47 &
  run 2 learnR-I "$CODE/learn_rotation.py" --device cuda:0 --init identity --out $ROOT/arms/learnedR-identity --steps ${LEARN_STEPS:-100} --layers 0-47 &
  run 3 learnR-H "$CODE/learn_rotation.py" --device cuda:0 --init had16 --out $ROOT/arms/learnedR-had16 --steps ${LEARN_STEPS:-100} --layers 0-47 &
  wait
  run 0 E1a "$CODE/run_exl3_anchor.py" --arm E1a --recon $ROOT/exl3-recon/4.0bpw --scope full --device cuda:0 --panels selection,final &
  run 1 B6i "$CODE/run_arm.py" --arm B6i --device cuda:0 --method gptq --rotation learned:$ROOT/arms/learnedR-identity --panels selection &
  run 2 B6h "$CODE/run_arm.py" --arm B6h --device cuda:0 --method gptq --rotation learned:$ROOT/arms/learnedR-had16 --panels selection &
  run 3 B4s4 "$CODE/run_arm.py" --arm B4s4 --device cuda:0 --method gptq --rotation random:4 --panels selection &
  wait
  log "phase C done"
  ;;
D)
  # allocation at E1b / E1a exact routed-expert bytes, then the allocated arms with final-role scoring
  B5=$($PY -c "import json;print(json.load(open('$ROOT/seals/bytes-exl3-5.0bpw.json'))['routed_expert_payload_bytes'])")
  B4=$($PY -c "import json;print(json.load(open('$ROOT/seals/bytes-exl3-4.0bpw.json'))['routed_expert_payload_bytes'])")
  log "budgets: E1b routed bytes=$B5  E1a routed bytes=$B4"
  $PY $CODE/allocate.py --candidates $ROOT/arms/cand-identity/candidates --out $ROOT/arms/alloc-E1b --budget-bytes $B5 --name E1b > $ROOT/logs/alloc-E1b.out 2>&1
  $PY $CODE/allocate.py --candidates $ROOT/arms/cand-identity/candidates --out $ROOT/arms/alloc-E1a --budget-bytes $B4 --name E1a > $ROOT/logs/alloc-E1a.out 2>&1
  log "alloc E1b: $(grep -E 'used_bpw|histogram' -A0 $ROOT/logs/alloc-E1b.out | tr '\n' ' ' | head -c 300)"
  run 0 B7  "$CODE/run_arm.py" --arm B7  --device cuda:0 --method gptq --tiermap $ROOT/arms/alloc-E1b/tiermap.json --panels selection,final,wikitext --a4 --save-weights &
  run 1 B7a "$CODE/run_arm.py" --arm B7a --device cuda:0 --method gptq --tiermap $ROOT/arms/alloc-E1a/tiermap.json --panels selection,final --a4 &
  run 2 B3p-final "$CODE/run_arm.py" --arm B3p-final --device cuda:0 --method gptq --panels final,wikitext --a4 &
  run 3 E1a-experts "$CODE/run_exl3_anchor.py" --arm E1a-experts --recon $ROOT/exl3-recon/4.0bpw --scope experts --device cuda:0 --panels selection,final &
  wait
  $PY $CODE/report.py --out $ROOT/REPORT.md > $ROOT/logs/report.out 2>&1
  log "phase D done"
  ;;
esac
