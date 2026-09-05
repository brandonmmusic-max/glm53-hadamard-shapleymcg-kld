#!/usr/bin/env bash
# Separate diagnostic only: the frozen speed series is never modified.
set -euo pipefail
REPO=${GLM53_REPO:-/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53}
RUNTIME=/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53-p8-all42-runtime
ROOT=/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/uniform-p8-all42-v1
SPEED=$ROOT/speed-v2a3
OUT=$ROOT/decode-profile-v1
UNIT=glm53-uniform-p8-speed-v2a3.service
CONTAINER=glm53-p8-decode-profile-v1
export PYTHONPATH="$REPO"

echo "Waiting for terminal speed series: $UNIT"
while true; do
  state=$(systemctl --user show "$UNIT" -p ActiveState --value)
  case "$state" in
    active|activating|deactivating|reloading) sleep 30 ;;
    inactive|failed) break ;;
    *) echo "Unexpected speed unit state: $state"; exit 1 ;;
  esac
done
[ "$(systemctl --user show "$UNIT" -p Result --value)" = success ]
[ -f "$SPEED/analysis.json" ]
(cd "$REPO/experiments" && sha256sum --check --strict p8-decode-profile-v1.sha256)
[ "$(git -C "$RUNTIME" rev-parse HEAD)" = efd250b002e8da5a0e603253e0cd07ba6249c7b4 ]
[ "$(sha256sum "$ROOT/manifest.json" | cut -d' ' -f1)" = 9521dae70165d5ba930ca447a37fef97801a6aa8638aef2417a4b206bdc8f191 ]
[ "$(sha256sum "$ROOT/runtime-patch-manifest-efd250b.json" | cut -d' ' -f1)" = e1cf316e8e969625eac6749866a12c298b5ae7fef3dde6ebba37b53ed74dc873 ]
python3 -m glm53_nvfp4.hash_tree --root "$RUNTIME/runtime_patch" \
  --output "$ROOT/runtime-patch-manifest-efd250b.json" --verify
python3 -m glm53_nvfp4.p8_profile_launch --root "$SPEED" \
  --plan "$REPO/experiments/p8-uniform-all42-tp4-vs-exl3-speed-v2.json" --output-dir "$OUT"

# Serialize with every production/research consumer. Never acquire this lock
# while waiting for the speed job, which owns it through its restore trap.
exec 9>/run/lock/klc/model-stack.lock
flock -w 900 9
[ "$(systemctl --user show "$UNIT" -p ActiveState --value)" = inactive ]
[ "$(systemctl --user show "$UNIT" -p Result --value)" = success ]
! docker inspect "$CONTAINER" >/dev/null 2>&1
[ -z "$(ss -H -ltn '( sport = :8018 )')" ]
timer_was_active=false
backend_was_active=false
systemctl is-active --quiet klc-model-stack.timer && timer_was_active=true
systemctl --user is-active --quiet klc-backend.service && backend_was_active=true
container_id=
thermal_pid=
cleanup_done=false

stop_owned() {
  [ "$cleanup_done" = true ] && return 0
  if [ -s "$OUT/container.cid" ]; then
    container_id=$(cat "$OUT/container.cid")
    [[ "$container_id" =~ ^[a-f0-9]{64}$ ]] || { echo 'Invalid owned CID file'; return 1; }
    # Allow bounded Nsight/application shutdown and report finalization.
    docker stop --time 120 "$container_id" >/dev/null 2>&1 || true
    docker logs "$container_id" >"$OUT/server-final.log" 2>&1 || true
    docker inspect "$container_id" >"$OUT/container-final.private.json" 2>/dev/null || true
    docker info >/dev/null 2>&1 || { echo 'Cannot establish Docker cleanup state'; return 1; }
    if [ "$(docker inspect -f '{{.State.Running}}' "$container_id" 2>/dev/null || true)" = true ]; then
      echo 'Diagnostic container did not stop; refusing to hide cleanup failure'
      return 1
    fi
    docker rm "$container_id" >/dev/null 2>&1 || true
  fi
  cleanup_done=true
}
restore() {
  rc=$?
  set +e
  if [ -n "$thermal_pid" ]; then kill "$thermal_pid" 2>/dev/null; wait "$thermal_pid" 2>/dev/null; fi
  cleanup_ok=true
  stop_owned || { rc=1; cleanup_ok=false; }
  if [ "$cleanup_ok" = true ]; then
    [ "$backend_was_active" = true ] && systemctl --user start klc-backend.service
    [ "$timer_was_active" = true ] && sudo -n systemctl start klc-model-stack.timer
  else
    echo 'Holding production restart because diagnostic cleanup failed'
  fi
  restored_timer=false
  restored_backend=false
  systemctl is-active --quiet klc-model-stack.timer && restored_timer=true
  systemctl --user is-active --quiet klc-backend.service && restored_backend=true
  [ "$restored_timer" = "$timer_was_active" ] && [ "$restored_backend" = "$backend_was_active" ] || rc=1
  python3 - "$OUT" "$rc" "$timer_was_active" "$backend_was_active" "$REPO" "$restored_timer" "$restored_backend" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
out, repo = Path(sys.argv[1]), Path(sys.argv[5])
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
record = {'schema':'p8-profile-execution.v1', 'exit_code':int(sys.argv[2]),
          'finished_at':datetime.now(timezone.utc).isoformat(),
          'prior_timer_active':sys.argv[3]=='true', 'prior_backend_active':sys.argv[4]=='true',
          'restored_timer_active':sys.argv[6]=='true', 'restored_backend_active':sys.argv[7]=='true',
          'diagnostic_plan_sha256':sha(repo/'experiments/p8-decode-profile-v1.json'),
          'script_sha256':sha(repo/'scripts/run_p8_decode_profile_followon.sh'),
          'client_sha256':sha(repo/'glm53_nvfp4/p8_profile_client.py'),
          'launch_generator_sha256':sha(repo/'glm53_nvfp4/p8_profile_launch.py'),
          'trace_verified':False, 'protected_roles_opened':[],
          'claim':'Diagnostic capture only; no speed/KLD qualification implied.'}
with (out/'execution.json').open('x') as f: json.dump(record,f,indent=2,sort_keys=True); f.write('\n')
PY
  [ "$?" -eq 0 ] || rc=1
  echo "Profile exit=$rc; restored timer=$restored_timer backend=$restored_backend"
  exit "$rc"
}
trap restore EXIT
[ "$timer_was_active" = true ] && sudo -n systemctl stop klc-model-stack.timer
[ "$backend_was_active" = true ] && systemctl --user stop klc-backend.service
deadline=$((SECONDS + 1800))
while true; do
  temps=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
  [ "$(printf '%s\n' "$temps" | wc -l)" -eq 4 ]
  ! printf '%s\n' "$temps" | grep -qvE '^[0-9]+$'
  max_temp=$(printf '%s\n' "$temps" | sort -nr | head -1)
  [ "$max_temp" -le 75 ] && break
  [ "$SECONDS" -lt "$deadline" ] || { echo 'Profile thermal start timeout'; exit 1; }
  sleep 30
done
nvidia-smi -q -x >"$OUT/nvidia-before.xml"
python3 - "$OUT/launch-spec.json" <<'PY'
import json, subprocess, sys
spec=json.load(open(sys.argv[1]))
if spec['status'] != 'prepared-not-launched': raise SystemExit('invalid launch spec')
subprocess.run(spec['argv'], check=True)
PY
container_id=$(cat "$OUT/container.cid")
[[ "$container_id" =~ ^[a-f0-9]{64}$ ]]
(
  telemetry_abort() {
    echo "$1" >"$OUT/thermal-abort.txt"
    docker stop --time 5 "$container_id" >/dev/null 2>&1 || true
    exit 1
  }
  while docker inspect -f '{{.State.Running}}' "$container_id" 2>/dev/null | grep -qx true; do
    temps=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits) || telemetry_abort 'GPU telemetry failed'
    [ "$(printf '%s\n' "$temps" | wc -l)" -eq 4 ] || telemetry_abort 'GPU telemetry inventory differs'
    if printf '%s\n' "$temps" | grep -qvE '^[0-9]+$'; then telemetry_abort 'GPU telemetry is not numeric'; fi
    max_temp=$(printf '%s\n' "$temps" | sort -nr | head -1)
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$max_temp" >>"$OUT/thermal-max.log"
    if [ "$max_temp" -ge 94 ]; then
      telemetry_abort "Thermal abort at $max_temp C"
    fi
    sleep 2
  done
) &
thermal_pid=$!
deadline=$((SECONDS + 2400))
until curl -fsS --max-time 5 http://127.0.0.1:8018/v1/models >"$OUT/models.json" 2>/dev/null && grep -q "$CONTAINER" "$OUT/models.json"; do
  [ "$(docker inspect -f '{{.State.Running}}' "$container_id" 2>/dev/null || true)" = true ]
  kill -0 "$thermal_pid"
  [ "$SECONDS" -lt "$deadline" ] || { echo 'Profile server readiness timeout'; exit 1; }
  sleep 5
done
docker logs "$container_id" >"$OUT/server-ready.log" 2>&1
python3 -m glm53_nvfp4.audit_speed_graphs --log "$OUT/server-ready.log"
python3 -m glm53_nvfp4.p8_profile_client --base-url http://127.0.0.1:8018 \
  --expected-model "$CONTAINER" --output-dir "$OUT/client" --prefill-iterations 16 \
  --profiler-delay-iterations 33 --profiler-max-iterations 16
[ ! -f "$OUT/thermal-abort.txt" ]
kill -0 "$thermal_pid"
stop_owned
nvidia-smi -q -x >"$OUT/nvidia-after.xml"
[ -s "$OUT/native-p8-c1.nsys-rep" ] || { echo 'Nsight report missing; preserve failed diagnostic'; exit 1; }
# Parse the report with the same in-image Nsight version, without GPU access.
docker run --rm --network none --entrypoint /usr/local/cuda/bin/nsys \
  -v "$OUT:/profile:rw" sha256:5da4ef3e814a71c6bcc47a7eb409a02fe4e3d5d867261f0b8e2e9b2d6ebb8ef8 \
  export --type sqlite --output /profile/native-p8-c1.sqlite /profile/native-p8-c1.nsys-rep \
  >"$OUT/export.log" 2>&1
python3 - "$OUT/native-p8-c1.sqlite" <<'PY'
import sqlite3, sys
with sqlite3.connect('file:'+sys.argv[1]+'?mode=ro', uri=True) as db:
    if db.execute('PRAGMA integrity_check').fetchone() != ('ok',):
        raise SystemExit('Nsight SQLite integrity check failed')
PY
echo "Parseable capture produced; rank/iteration coverage still requires audit: $OUT"
