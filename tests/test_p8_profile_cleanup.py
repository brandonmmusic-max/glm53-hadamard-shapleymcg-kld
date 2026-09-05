"""Exercise the actual shell cleanup function against CPU-only Docker mocks."""
import json
import os
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_p8_decode_profile_followon.sh'
CID = 'a' * 64


def run_cleanup(tmp_path, *, cid=CID, mode='stopped', twice=False):
    out = tmp_path / 'out'
    out.mkdir()
    if cid is not None:
        (out / 'container.cid').write_text(cid)
    mock = tmp_path / 'docker'
    mock.write_text('''#!/usr/bin/python3
import json, os, sys
with open(os.environ['P8_TEST_LOG'], 'a') as f:
    f.write(json.dumps(sys.argv[1:])+'\\n')
mode=os.environ['P8_TEST_MODE']
if sys.argv[1]=='info' and mode=='unreachable': sys.exit(1)
if sys.argv[1]=='inspect':
    print('true' if mode=='running' else 'false')
if sys.argv[1]=='logs': print('original diagnostic log')
''')
    mock.chmod(0o755)
    source = SCRIPT.read_text()
    function = source[source.index('stop_owned() {'):source.index('restore() {')]
    harness = tmp_path / 'cleanup.sh'
    harness.write_text('set -euo pipefail\nOUT="$P8_TEST_OUT"\ncleanup_done=false\n'
                       + function + '\nstop_owned\n' + ('stop_owned\n' if twice else ''))
    env = {**os.environ, 'PATH': f'{tmp_path}:/usr/bin:/bin',
           'P8_TEST_OUT': str(out), 'P8_TEST_LOG': str(tmp_path / 'calls.jsonl'),
           'P8_TEST_MODE': mode}
    result = subprocess.run(['/bin/bash', str(harness)], env=env,
                            capture_output=True, text=True, timeout=5)
    log = tmp_path / 'calls.jsonl'
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls, out


def test_missing_cid_never_touches_a_container(tmp_path):
    result, calls, _ = run_cleanup(tmp_path, cid=None)
    assert result.returncode == 0
    assert calls == []


@pytest.mark.parametrize('cid', ['glm53-p8-decode-profile-v1', 'not-a-container', 'a' * 63])
def test_invalid_cid_fails_without_docker_actions(tmp_path, cid):
    result, calls, _ = run_cleanup(tmp_path, cid=cid)
    assert result.returncode != 0
    assert calls == []


def test_cleanup_targets_cid_and_preserves_log_on_second_call(tmp_path):
    result, calls, out = run_cleanup(tmp_path, twice=True)
    assert result.returncode == 0
    assert len([args for args in calls if args[0] == 'stop']) == 1
    assert len([args for args in calls if args[0] == 'logs']) == 1
    assert (out / 'server-final.log').read_text() == 'original diagnostic log\n'
    for args in calls:
        if args[0] != 'info':
            assert args[-1] == CID


@pytest.mark.parametrize('mode', ['running', 'unreachable'])
def test_uncertain_or_running_container_blocks_success(tmp_path, mode):
    result, calls, _ = run_cleanup(tmp_path, mode=mode)
    assert result.returncode != 0
    assert not any(args[0] == 'rm' for args in calls)
