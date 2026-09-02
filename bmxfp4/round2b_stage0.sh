#!/usr/bin/env bash
# Round 2b stage 0 (REAP calibration corpus): teacher logits for the 32 attribution windows (gpu0), then
# Hessians + routed samples on all 64 calibration windows, 12 layers per GPU, into hessians-calib/.
# gpu2/gpu3 wait for the (off-plan, window-calibrated) B7u/B7ua processes to exit first.
set -u
ROOT=/media/brandonmusic/klcstore/bmxfp4; L=$ROOT/logs
PY=/home/brandonmusic/klc-env/bin/python; CODE=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-qwen3-30b-a3b/bmxfp4
export BMXFP4_HESS_DIR=$ROOT/hessians-calib
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/campaign.log; }
cd $CODE
log "ROUND 2b stage 0: REAP-calibration teacher (calib-attrib, gpu0) + Hessians (calib 64 windows) -> hessians-calib"
( CUDA_VISIBLE_DEVICES=0 $PY run_teacher.py --device cuda:0 --panels calib-attrib --b0-runs 1 > $L/teacher-calib.out 2>&1; log "teacher calib-attrib rc=$?";
  CUDA_VISIBLE_DEVICES=0 $PY run_hessians.py --device cuda:0 --layers 0-11 --role calib > $L/hess-calib-0-11.out 2>&1; log "hessians 0-11 rc=$?" ) &
( CUDA_VISIBLE_DEVICES=1 $PY run_hessians.py --device cuda:0 --layers 12-23 --role calib > $L/hess-calib-12-23.out 2>&1; log "hessians 12-23 rc=$?" ) &
( while kill -0 3740736 2>/dev/null; do sleep 5; done; CUDA_VISIBLE_DEVICES=2 $PY run_hessians.py --device cuda:0 --layers 24-35 --role calib > $L/hess-calib-24-35.out 2>&1; log "hessians 24-35 rc=$?" ) &
( while kill -0 3740735 2>/dev/null; do sleep 5; done; CUDA_VISIBLE_DEVICES=3 $PY run_hessians.py --device cuda:0 --layers 36-47 --role calib > $L/hess-calib-36-47.out 2>&1; log "hessians 36-47 rc=$?" ) &
wait
log "ROUND 2b stage 0 done: $(ls $ROOT/hessians-calib/layer-*.safetensors 2>/dev/null | wc -l) layers, $(ls $ROOT/teacher/calib-attrib/row-*.safetensors 2>/dev/null | wc -l) teacher rows"
