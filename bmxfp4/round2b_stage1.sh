#!/usr/bin/env bash
# Round 2b stage 1 (REAP calibration corpus, hessians-calib):
#   gpu0: B5-prov2 (uniform NVFP4, Had16, GPTQ; provisional endpoint, saved weights)  -> attribution half A (calib-attrib 0-15)
#   gpu1: cand-calib-had16 (4:8 T0 ladder, Had16 basis, calib samples)              -> attribution half B (calib-attrib 16-31)
#   gpu2: learn_rotation_v2 layers 0-23  -> B5L2 (selection, learned basis)  [basis decision, DECISIONS #9]
#   gpu3: learn_rotation_v2 layers 24-47
# then: merge attribution -> calibrated allocations B7/B7a + uncalibrated B7u/B7ua -> causal re-encode x4 (one per GPU)
set -u
ROOT=/media/brandonmusic/klcstore/bmxfp4; L=$ROOT/logs; ARMS=$ROOT/arms
PY=/home/brandonmusic/klc-env/bin/python; CODE=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-qwen3-30b-a3b/bmxfp4
export BMXFP4_HESS_DIR=$ROOT/hessians-calib
E1B=$(python3 -c "import json;print(json.load(open('$ROOT/seals/bytes-exl3-5.0bpw.json'))['routed_expert_payload_bytes'])")
E1A=$(python3 -c "import json;print(json.load(open('$ROOT/seals/bytes-exl3-4.0bpw.json'))['routed_expert_payload_bytes'])")
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/campaign.log; }
cd $CODE
log "ROUND 2b stage 1 start (hessians-calib layers: $(ls $ROOT/hessians-calib/layer-*.safetensors | wc -l))"
( CUDA_VISIBLE_DEVICES=0 $PY run_arm.py --arm B5-prov2 --device cuda:0 --method gptq --rotation had16 --panels selection --save-weights > $L/B5-prov2.out 2>&1; log "B5-prov2 rc=$? $(tail -n1 $L/arm-B5-prov2.log)";
  CUDA_VISIBLE_DEVICES=0 $PY attribution.py --endpoint $ARMS/B5-prov2 --device cuda:0 --role calib-attrib --windows 0-15 --chunk 8 --out $ARMS/attrib-B5prov2/part-0.json > $L/attrib2-a.out 2>&1; log "attrib2-a rc=$?" ) &
( CUDA_VISIBLE_DEVICES=1 $PY run_arm.py --arm cand-calib-had16 --device cuda:0 --mode candidates --method gptq --rotation had16 > $L/cand-calib-had16.out 2>&1; log "cand-calib-had16 rc=$? layers=$(ls $ARMS/cand-calib-had16/candidates | wc -l)";
  while [ ! -f $ARMS/B5-prov2/weights-layer-047.safetensors ] || [ ! -f $ARMS/B5-prov2/receipt.json ]; do sleep 10; done
  CUDA_VISIBLE_DEVICES=1 $PY attribution.py --endpoint $ARMS/B5-prov2 --device cuda:0 --role calib-attrib --windows 16-31 --chunk 8 --out $ARMS/attrib-B5prov2/part-1.json > $L/attrib2-b.out 2>&1; log "attrib2-b rc=$?" ) &
( CUDA_VISIBLE_DEVICES=2 $PY learn_rotation_v2.py --device cuda:0 --layers 0-23 --out $ARMS/learnedR-cal2 --steps 80 > $L/learnR2b-a.out 2>&1; log "learnR2b 0-23 rc=$?" ) &
( CUDA_VISIBLE_DEVICES=3 $PY learn_rotation_v2.py --device cuda:0 --layers 24-47 --out $ARMS/learnedR-cal2 --steps 80 > $L/learnR2b-b.out 2>&1; log "learnR2b 24-47 rc=$?" ) &
wait
CUDA_VISIBLE_DEVICES=2 $PY run_arm.py --arm B5L2 --device cuda:0 --method gptq --rotation learned:$ARMS/learnedR-cal2 --panels selection > $L/B5L2.out 2>&1; log "B5L2 rc=$? $(tail -n1 $L/arm-B5L2.log)"
$PY attribution.py --merge $ARMS/attrib-B5prov2 --out $ARMS/attrib-B5prov2/attribution.json > $L/attrib2-merge.out 2>&1; log "attribution merge: $(grep -E 'kld_end|sum_attribution|remainder' $L/attrib2-merge.out | tr -d '\n ')"
CAND=$ARMS/cand-calib-had16/candidates
$PY allocate.py --candidates $CAND --out $ARMS/alloc-B7  --budget-bytes $E1B --name B7  --attribution $ARMS/attrib-B5prov2/attribution.json > $L/alloc-B7.out 2>&1  || log "alloc B7 FAILED"
$PY allocate.py --candidates $CAND --out $ARMS/alloc-B7a --budget-bytes $E1A --name B7a --attribution $ARMS/attrib-B5prov2/attribution.json > $L/alloc-B7a.out 2>&1 || log "alloc B7a FAILED"
$PY allocate.py --candidates $CAND --out $ARMS/alloc-B7u  --budget-bytes $E1B --name B7u  > $L/alloc-B7u.out 2>&1  || log "alloc B7u FAILED"
$PY allocate.py --candidates $CAND --out $ARMS/alloc-B7ua --budget-bytes $E1A --name B7ua > $L/alloc-B7ua.out 2>&1 || log "alloc B7ua FAILED"
for a in B7 B7a B7u B7ua; do log "alloc $a: $(python3 -c "import json;d=json.load(open('$ARMS/alloc-$a/summary.json'));print({t.split('_')[0]:v['units'] for t,v in d['histogram'].items()}, 'used_bpw=%.4f'%d['used_bpw'], 'stats', {k:v for k,v in (d.get('attribution_stats') or {}).items() if k!='layer_median_scale'})")"; done
log "START causal (had16 basis, calib Hessians): B7 gpu0, B7a gpu1, B7u gpu2, B7ua gpu3"
CUDA_VISIBLE_DEVICES=0 $PY run_causal.py --arm B7   --device cuda:0 --rotation had16 --tiermap $ARMS/alloc-B7/tiermap.json   --calib-role calib-attrib --panels selection,final,wikitext --a4 --save-weights > $L/B7.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY run_causal.py --arm B7a  --device cuda:0 --rotation had16 --tiermap $ARMS/alloc-B7a/tiermap.json  --calib-role calib-attrib --panels selection,final,wikitext --a4 --save-weights > $L/B7a.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY run_causal.py --arm B7u  --device cuda:0 --rotation had16 --tiermap $ARMS/alloc-B7u/tiermap.json  --calib-role calib-attrib --panels selection,final,wikitext --a4 > $L/B7u.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY run_causal.py --arm B7ua --device cuda:0 --rotation had16 --tiermap $ARMS/alloc-B7ua/tiermap.json --calib-role calib-attrib --panels selection,final,wikitext --a4 > $L/B7ua.out 2>&1 &
wait
for a in B7 B7a B7u B7ua; do log "$a: $(tail -n1 $L/arm-$a.log)"; done
log "ROUND 2b stage 1 done"
