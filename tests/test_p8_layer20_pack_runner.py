import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def fixture(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'scripts'))
    spec = importlib.util.spec_from_file_location('pack_runner', root / 'scripts/run_p8_layer20_pack.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path / 'repo'
    module.OUT = tmp_path / 'artifacts'
    module.OUT.mkdir()
    (module.ROOT / 'experiments').mkdir(parents=True)
    plan = dict(layer=20,campaign_aggregate_bound_bytes=25753000000,user_ceiling_bytes=30000000000,existing_external_bound_bytes=9643000000,packer_sha256='pin',verifier_sha256='pin',source_ranges=[[0,72],[72,144],[144,216],[216,288]],layer_artifact_bound_bytes=16110000000)
    (module.ROOT / 'experiments/p8-coupled-layer20-pack-v1.json').write_text(json.dumps(plan))
    monkeypatch.setattr(module,'sha',lambda _: 'pin')
    monkeypatch.setattr(module,'validate_chunk',lambda a,b,layer: f'{a}:{b}')
    monkeypatch.setattr(module,'paths',lambda a,b,layer: (tmp_path / f'{a}-{b}.safetensors',tmp_path / f'{a}-{b}.json'))
    monkeypatch.setattr(module.shutil,'disk_usage',lambda _: SimpleNamespace(free=10000000000))
    monkeypatch.setattr(sys,'argv',['pack_runner','--execute'])
    return module


def test_failed_packer_does_not_run_verifier(monkeypatch,tmp_path):
    module = fixture(monkeypatch,tmp_path)
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        return SimpleNamespace(returncode=9)
    monkeypatch.setattr(module.subprocess,'run',fail)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 9
    assert len(calls) == 1
    assert json.loads((module.OUT/'layer20-pack.execution.json').read_text())['exit_code'] == 9
    assert not (module.OUT/'layer20-postwrite.launch.json').exists()


def test_missing_prior_chunk_blocks_before_writes(monkeypatch,tmp_path):
    module = fixture(monkeypatch,tmp_path)
    def missing(a,b,layer):
        raise FileNotFoundError('missing prior')
    monkeypatch.setattr(module,'validate_chunk',missing)
    with pytest.raises(FileNotFoundError):
        module.main()
    assert list(module.OUT.iterdir()) == []


def test_existing_target_is_preserved(monkeypatch,tmp_path):
    module = fixture(monkeypatch,tmp_path)
    target = module.OUT/'sidecars/layer-020'
    target.mkdir(parents=True)
    saved = target/'retained'
    saved.write_bytes(b'preserve')
    with pytest.raises(AssertionError):
        module.main()
    assert saved.read_bytes() == b'preserve'


def test_layer20_commands_and_budget_gate(monkeypatch,tmp_path):
    module = fixture(monkeypatch,tmp_path)
    seen = []
    def validate(a,b,layer):
        assert layer == 20
        return f'{a}:{b}'
    monkeypatch.setattr(module,'validate_chunk',validate)
    def launch(command,**kwargs):
        seen.append(command)
        assert command[command.index('--layer')+1] == '20'
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        return SimpleNamespace(returncode=9)
    monkeypatch.setattr(module.subprocess,'run',launch)
    with pytest.raises(SystemExit):
        module.main()
    assert len(seen) == 1


def test_over_budget_stops_before_launch(monkeypatch,tmp_path):
    module = fixture(monkeypatch,tmp_path)
    path = module.ROOT/'experiments/p8-coupled-layer20-pack-v1.json'
    plan = json.loads(path.read_text())
    plan['campaign_aggregate_bound_bytes'] = 30000000001
    path.write_text(json.dumps(plan))
    with pytest.raises(AssertionError):
        module.main()
    assert list(module.OUT.iterdir()) == []
