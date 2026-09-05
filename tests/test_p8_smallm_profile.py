import copy
import json
import shlex
from pathlib import Path

import pytest

from glm53_nvfp4.p8_smallm_profile import build_argv, IMAGE, NAME, PORT

SOURCE = Path(__file__).resolve().parents[1] / 'evidence/opened/codec-v2/p8-smallm-integrated-v1/container.json'


def test_profile_preserves_measured_serving_configuration():
    source = json.loads(SOURCE.read_text())[0]
    argv = build_argv(source, Path('/tmp/p8-profile-test'))
    original = shlex.split(source['Config']['Cmd'][1])[1:]
    start = argv.index('/opt/venv/bin/python')
    profiled = argv[start:]
    for option in ('--port', '--served-model-name'):
        idx = profiled.index(option)
        profiled[idx+1] = original[original.index(option)+1]
    assert profiled[:-2] == original
    assert argv[argv.index('--entrypoint')+1] == '/usr/local/cuda/bin/nsys'
    assert IMAGE in argv and '--cuda-graph-trace=node' in argv
    for env in source['Config']['Env']:
        assert env in argv
    for mount in source['HostConfig']['Binds']:
        assert mount in argv


def test_profile_rejects_drifted_runtime():
    source = json.loads(SOURCE.read_text())[0]
    changed = copy.deepcopy(source)
    changed['Config']['Cmd'][1] += ' --enforce-eager'
    with pytest.raises(ValueError):
        build_argv(changed, Path('/tmp/p8-profile-test'))
    changed = copy.deepcopy(source)
    changed['Image'] = 'different'
    with pytest.raises(ValueError):
        build_argv(changed, Path('/tmp/p8-profile-test'))
