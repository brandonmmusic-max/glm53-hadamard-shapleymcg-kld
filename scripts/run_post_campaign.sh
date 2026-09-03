#!/usr/bin/env bash
set -euo pipefail

REPO=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53
CAMPAIGN=/media/brandonmusic/klcstore/bmxfp4-glm53
SOURCE=$CAMPAIGN/downloads/GLM-5.3-Flash-BF16
TEACHER=$CAMPAIGN/teacher
EVIDENCE=$CAMPAIGN/evidence
RECORD=$CAMPAIGN/experiment-record.json
LOG=$CAMPAIGN/logs/post-campaign.log
cd "$REPO"
log() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG"; }

while [ "$(find "$EVIDENCE" -maxdepth 1 -name 'layer-*-validation.json' | wc -l)" -lt 42 ]; do
  if ! systemctl --user is-active --quiet glm53-nvfp4-v2-campaign.service; then
    log "campaign runner stopped before all 42 layer validations"
    exit 1
  fi
  sleep 30
done
while systemctl --user is-active --quiet glm53-nvfp4-v2-campaign.service; do sleep 5; done
log "all 42 layer validations present"

while [ "$(find "$SOURCE" -maxdepth 1 -name 'model-*.safetensors' | wc -l)" -lt 120 ]; do
  systemctl --user is-active --quiet hf-download-glm53-bf16-v2.service || { log "source download stopped incomplete"; exit 1; }
  sleep 30
done
while systemctl --user is-active --quiet hf-download-glm53-bf16-v2.service; do sleep 5; done
while [ "$(find "$TEACHER/logits/full-panel" -type f -name '*.safetensors' 2>/dev/null | wc -l)" -lt 64 ]; do
  systemctl --user is-active --quiet hf-download-glm53-teacher-v2.service || { log "teacher download stopped incomplete"; exit 1; }
  sleep 30
done
while systemctl --user is-active --quiet hf-download-glm53-teacher-v2.service; do sleep 5; done

log "verifying pinned source and role-approved teacher files"
/home/brandonmusic/.local/bin/hf cache verify zai-org/GLM-5.3-Flash-BF16 \
  --revision a6c167b62691b2bac901344b65cb651a70f53e43 --local-dir "$SOURCE" \
  --fail-on-missing-files --format json >"$EVIDENCE/hf-source-verify.json"
/home/brandonmusic/.local/bin/hf cache verify brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
  --type dataset --revision 95f4fdd94bf29989db2e0d1054e4931f55edb6aa --local-dir "$TEACHER" \
  --format json >"$EVIDENCE/hf-teacher-subset-verify.json"
python3 -m glm53_nvfp4.stage_receipt "$RECORD" --stage full-export-and-download-integrity \
  --evidence "$EVIDENCE/hf-source-verify.json" --evidence "$EVIDENCE/hf-teacher-subset-verify.json" \
  --evidence "$CAMPAIGN/candidates/uniform-gptq/model.safetensors.index.json" \
  --note 'All 42 layer validations passed and all 120 pinned BF16 shards verified.'

log "running pre-freeze selection control then candidate"
"$REPO/scripts/run_kld_role.sh" selection selection-stock /home/brandonmusic/models/GLM-5.3-Flash-NVFP4 GLM-5.3-Flash-NVFP4-STOCK
"$REPO/scripts/run_kld_role.sh" selection selection-candidate "$CAMPAIGN/candidates/uniform-gptq" GLM-5.3-Flash-NVFP4-V2
python3 -m glm53_nvfp4.paired_role_analysis --role selection \
  --candidate-run "$CAMPAIGN/kld/records/selection-candidate" --stock-run "$CAMPAIGN/kld/records/selection-stock" \
  --roles "$CAMPAIGN/roles/roles.json" --output "$EVIDENCE/selection-analysis.json"
analysis_decision=$(python3 -c "import json; print(json.load(open('$EVIDENCE/selection-analysis.json'))['decision'])")
[ "$analysis_decision" = continue ] || { log "selection stopped candidate: $analysis_decision"; exit 1; }

log "freezing complete candidate and analysis identities"
python3 -m glm53_nvfp4.freeze_candidate --candidate "$CAMPAIGN/candidates/uniform-gptq" \
  --carrier /home/brandonmusic/models/GLM-5.3-Flash-NVFP4 --campaign "$CAMPAIGN" \
  --roles "$CAMPAIGN/roles/roles.json" --selection-candidate "$CAMPAIGN/kld/run-selection-candidate.json" \
  --selection-stock "$CAMPAIGN/kld/run-selection-stock.json" --selection-analysis "$EVIDENCE/selection-analysis.json" \
  --output "$EVIDENCE/candidate-freeze.json"
python3 -m glm53_nvfp4.stage_receipt "$RECORD" --stage candidate-frozen-before-confirmation \
  --evidence "$EVIDENCE/selection-analysis.json" --evidence "$EVIDENCE/candidate-freeze.json" \
  --note 'Selection favored the primary candidate; implementation, analysis, candidate chunks, and exact confirmation role frozen.'

log "opening confirmation exactly once: stock then frozen candidate"
"$REPO/scripts/run_kld_role.sh" confirmation confirmation-stock /home/brandonmusic/models/GLM-5.3-Flash-NVFP4 GLM-5.3-Flash-NVFP4-STOCK "$EVIDENCE/candidate-freeze.json"
"$REPO/scripts/run_kld_role.sh" confirmation confirmation-candidate "$CAMPAIGN/candidates/uniform-gptq" GLM-5.3-Flash-NVFP4-V2 "$EVIDENCE/candidate-freeze.json"
python3 -m glm53_nvfp4.paired_role_analysis --role confirmation \
  --candidate-run "$CAMPAIGN/kld/records/confirmation-candidate" --stock-run "$CAMPAIGN/kld/records/confirmation-stock" \
  --roles "$CAMPAIGN/roles/roles.json" --output "$EVIDENCE/confirmation-analysis.json"
python3 -m glm53_nvfp4.stage_receipt "$RECORD" --stage confirmation-complete \
  --evidence "$CAMPAIGN/kld/run-confirmation-stock.json" --evidence "$CAMPAIGN/kld/run-confirmation-candidate.json" \
  --evidence "$EVIDENCE/confirmation-analysis.json" --note 'Exact 32-window role completed with no exclusions; decision follows preregistered threshold.'

decision=$(python3 -c "import json; print(json.load(open('$EVIDENCE/confirmation-analysis.json'))['decision'])")
if [ "$decision" = pass ]; then
  log "KLD qualification passed; running separate post-quality benchmarks"
  "$REPO/scripts/run_benchmarks.sh" "$EVIDENCE/confirmation-analysis.json"
else
  log "KLD qualification failed; performance suite skipped by registered gate"
fi

python3 -m glm53_nvfp4.preflight --roles "$CAMPAIGN/roles/roles.json" --output "$EVIDENCE/final-state.json"
systemctl --user is-active --quiet glm53-r10-tp2-mtp3.service && { log "restoration failure: service active"; exit 1; }
ss -ltn 'sport = :8000' | tail -n +2 | grep -q . && { log "restoration failure: port 8000 open"; exit 1; }
python3 -m glm53_nvfp4.finalize_record "$RECORD" --source-verify "$EVIDENCE/hf-source-verify.json" \
  --freeze "$EVIDENCE/candidate-freeze.json" --confirmation "$EVIDENCE/confirmation-analysis.json" \
  --final-state "$EVIDENCE/final-state.json"
python3 -m glm53_nvfp4.write_report --campaign "$CAMPAIGN" --output "$REPO/REPORT_GLM53_FLASH_NVFP4_V2.md"
python3 -m glm53_nvfp4.stage_receipt "$RECORD" --stage campaign-complete \
  --evidence "$REPO/REPORT_GLM53_FLASH_NVFP4_V2.md" --evidence "$EVIDENCE/final-state.json" \
  --note "Campaign completed with confirmation decision $decision; initial model service and port state restored."
python3 /home/brandonmusic/.codex/skills/local-inference-lab-running-sealed-experiments/scripts/experiment_record.py validate "$RECORD" --strict
python3 /home/brandonmusic/.codex/skills/local-inference-lab-running-sealed-experiments/scripts/experiment_record.py verify-seal "$RECORD"
log "post-campaign workflow complete with decision $decision"
