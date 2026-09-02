#!/usr/bin/env bash
# Round 2 (ShapleyMCG path): merge attribution -> calibrated allocation at the two exact budgets -> causal re-encode arms.
#   usage: round2_chain.sh ROT CAND_DIR          e.g. round2_chain.sh had16 /media/.../arms/cand48-had16/candidates
# Arms: B7  = calibrated (Shapley) allocation @ E1b bytes (5.029 bpw)      gpu0
#       B7a = calibrated (Shapley) allocation @ E1a bytes (4.029 bpw)      gpu1
#       B7u = uncalibrated proxy allocation  @ E1b bytes (control)         gpu2
set -u
ROT=${1:?rotation spec}; CAND=${2:?candidates dir}
ROOT=/media/brandonmusic/klcstore/bmxfp4; L=$ROOT/logs; ARMS=$ROOT/arms
PY=/home/brandonmusic/klc-env/bin/python; CODE=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-qwen3-30b-a3b/bmxfp4
E1B=$(python3 -c "import json;print(json.load(open('$ROOT/seals/bytes-exl3-5.0bpw.json'))['routed_expert_payload_bytes'])")
E1A=$(python3 -c "import json;print(json.load(open('$ROOT/seals/bytes-exl3-4.0bpw.json'))['routed_expert_payload_bytes'])")
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/campaign.log; }
cd $CODE
log "ROUND 2 chain: rot=$ROT cand=$CAND budgets E1b=$E1B E1a=$E1A"
$PY attribution.py --merge $ARMS/attrib-B5prov --out $ARMS/attrib-B5prov/attribution.json > $L/attrib-merge.out 2>&1 || { log "merge FAILED"; exit 1; }
grep -E "kld_end|sum_attribution|remainder|windows" $L/attrib-merge.out | tr -d '\n' | tee -a $L/campaign.log; echo
$PY allocate.py --candidates $CAND --out $ARMS/alloc-B7 --budget-bytes $E1B --name B7 --attribution $ARMS/attrib-B5prov/attribution.json > $L/alloc-B7.out 2>&1 || { log "alloc B7 FAILED"; exit 1; }
$PY allocate.py --candidates $CAND --out $ARMS/alloc-B7a --budget-bytes $E1A --name B7a --attribution $ARMS/attrib-B5prov/attribution.json > $L/alloc-B7a.out 2>&1 || { log "alloc B7a FAILED"; exit 1; }
$PY allocate.py --candidates $CAND --out $ARMS/alloc-B7u --budget-bytes $E1B --name B7u > $L/alloc-B7u.out 2>&1 || { log "alloc B7u FAILED"; exit 1; }
for a in B7 B7a B7u; do log "alloc $a: $(python3 -c "import json;d=json.load(open('$ARMS/alloc-$a/summary.json'));print({t:v['units'] for t,v in d['histogram'].items()}, 'used_bpw=%.4f'%d['used_bpw'])")"; done
log "START causal B7 (gpu0) B7a (gpu1) B7u (gpu2)"
CUDA_VISIBLE_DEVICES=0 $PY run_causal.py --arm B7  --device cuda:0 --rotation $ROT --tiermap $ARMS/alloc-B7/tiermap.json  --panels selection,final,wikitext --a4 --save-weights > $L/B7.out 2>&1 &  P0=$!
CUDA_VISIBLE_DEVICES=1 $PY run_causal.py --arm B7a --device cuda:0 --rotation $ROT --tiermap $ARMS/alloc-B7a/tiermap.json --panels selection,final,wikitext --a4 --save-weights > $L/B7a.out 2>&1 & P1=$!
CUDA_VISIBLE_DEVICES=2 $PY run_causal.py --arm B7u --device cuda:0 --rotation $ROT --tiermap $ARMS/alloc-B7u/tiermap.json --panels selection,final --a4 > $L/B7u.out 2>&1 & P2=$!
wait $P0; log "END B7 rc=$?";  wait $P1; log "END B7a rc=$?";  wait $P2; log "END B7u rc=$?"
for a in B7 B7a B7u; do tail -n 1 $L/arm-$a.log; done
log "ROUND 2 chain done"
