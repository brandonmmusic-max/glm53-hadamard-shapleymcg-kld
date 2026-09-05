import pytest

from glm53_nvfp4 import p8_weight_identity as identity


def test_actual_bytes_and_stat_are_bound(tmp_path):
    path = tmp_path / 'weight.safetensors'
    path.write_bytes(b'payload')
    row = {'arm': 'p8', 'path': str(path), 'expected_sha256': identity.sha(path),
           'expected_bytes': 7}
    result = identity.audit_one(row)
    assert result['stat']['bytes'] == 7
    assert result['sha256'] == row['expected_sha256']
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs'):
        identity.audit_one(row)


def test_stat_drift_and_inventory_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / 'weight.safetensors'
    path.write_bytes(b'payload')
    entry = {'path': str(path), 'stat': identity.fingerprint(path),
             'sha256': identity.sha(path), 'expected_sha256': identity.sha(path)}
    fingerprint = identity.fingerprint
    monkeypatch.setattr(identity, 'fingerprint', lambda ignored: fingerprint(path))
    receipt = {'schema': 'glm53-p8-exl3-fresh-payload-identity.v1', 'status': 'pass',
               'counts': {'p8': 168, 'exl3': 120},
               'files': [{**entry, 'path': str(tmp_path / str(i)),
                          'arm': 'p8' if i < 168 else 'exl3'} for i in range(288)]}
    identity.verify_stats(receipt)
    path.write_bytes(b'longer payload')
    with pytest.raises(ValueError, match='changed'):
        identity.verify_stats(receipt)
    receipt['files'] = []
    with pytest.raises(ValueError, match='inventory'):
        identity.verify_stats(receipt)


def test_size_mismatch_does_not_pass_with_matching_hash(tmp_path):
    path = tmp_path / 'weight.safetensors'
    path.write_bytes(b'payload')
    with pytest.raises(ValueError):
        identity.audit_one({'path': str(path), 'expected_sha256': identity.sha(path),
                            'expected_bytes': 8})
