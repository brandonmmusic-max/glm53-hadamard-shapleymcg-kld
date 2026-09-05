import importlib.util
import json
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_p8_layer3_remainder.py'
    spec = importlib.util.spec_from_file_location('remainder', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prior_chunk_hash_and_design_must_validate(tmp_path):
    module = load()
    module.OUT = tmp_path
    codec, receipt = module.paths(0,72)
    codec.parent.mkdir(parents=True)
    receipt.parent.mkdir(parents=True)
    codec.write_bytes(b'fixture')
    item = dict(schema='glm53-hessian-trellis-p8-coupled-scale-layer-chunk-receipt.v1',layer=3,expert_range=[0,72],design_sha256=module.DESIGN,protected_roles_opened=[],calibration=dict(role='fit',samples=256),algorithm=dict(ldlq=False,intermediate_draws=[0]*72),outputs=dict(codec=dict(path=str(codec),bytes=7,sha256=module.sha(codec))))
    receipt.write_text(json.dumps(item))
    assert module.validate_chunk(0,72) == module.sha(receipt)
    codec.write_bytes(b'corrupt')
    with pytest.raises(AssertionError):
        module.validate_chunk(0,72)
    codec.write_bytes(b'fixture')
    item['design_sha256'] = '0'*64
    receipt.write_text(json.dumps(item))
    with pytest.raises(AssertionError):
        module.validate_chunk(0,72)


def test_argument_replacement_does_not_mutate_other_inputs():
    module = load()
    command = ['python','--expert-start','0','--layer','3']
    module.replace_arg(command,'--expert-start',72)
    assert command == ['python','--expert-start','72','--layer','3']
