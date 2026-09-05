import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from glm53_nvfp4 import p8_decode_analysis as analysis


def records():
    return [{'window_id': f'conditional-fit-{i:04d}',
             'domain': sorted(analysis.protocol.DOMAINS)[i // 8],
             'rows': 2047, 'n128_mean_kld': .01 + i * .001,
             'n64_mean_kld': .01 + i * .001} for i in range(32)]


def test_exact_closure_reports_identity_not_a_fabricated_zero_width_ci():
    result = analysis.aggregate(records(), True)
    assert result['decision'] == 'exact-closure-pass'
    assert result['causal_rows'] == 65504
    delta = result['paired_candidate_minus_reference']
    assert delta['mean_delta_kld'] == 0
    assert delta['ci95_bca'] is None
    assert 'undefined' in delta['ci_status']
    assert result['reference_n128'] == result['candidate_n64']
    assert result['candidate_n64']['ci95_bca'][0] < result['candidate_n64']['mean_kld']
    assert result['candidate_n64']['ci95_bca'][1] > result['candidate_n64']['mean_kld']


def test_kld_improvement_cannot_replace_failed_exact_gate():
    rows = records()
    for row in rows:
        row['n64_mean_kld'] *= .8
    result = analysis.aggregate(rows, False)
    assert result['decision'] == 'exact-closure-fail'
    assert result['paired_candidate_minus_reference']['ci95_bca'][1] < 0
    assert result['paired_candidate_minus_reference']['relative_kld_reduction'] == pytest.approx(.2)
    with pytest.raises(ValueError, match='different KLD'):
        analysis.aggregate(rows, True)


@pytest.mark.parametrize('mutation', [
    lambda r: r.pop(),
    lambda r: r[1].update(window_id=r[0]['window_id']),
    lambda r: r[0].update(domain='protected'),
    lambda r: r[0].update(rows=2046),
    lambda r: r[0].update(n64_mean_kld=float('nan')),
])
def test_missing_misaligned_unbalanced_or_nonfinite_panel_rejected(mutation):
    rows = records()
    mutation(rows)
    with pytest.raises(ValueError):
        analysis.aggregate(rows, False)


def test_paired_bootstrap_is_reproducible():
    rows = records()
    for i, row in enumerate(rows):
        row['n64_mean_kld'] += (i % 5 - 2) * .001
    assert analysis.aggregate(rows, False) == analysis.aggregate(copy.deepcopy(rows), False)


def test_exact_score_reuse_requires_declared_verified_identity(monkeypatch):
    calls = []
    def score(teacher, student, tokens):
        calls.append(student)
        return SimpleNamespace(teacher_top1=np.array([1, 2, 3]), realized_token=np.array([1, 2, 4]))
    monkeypatch.setattr(analysis.protocol, 'score_aligned', score)
    a, b, agreement = analysis.score_capture_pair(None, 'a', 'b', None, True)
    assert a is b and calls == ['a'] and agreement == pytest.approx(2 / 3)
    calls.clear()
    a, b, _ = analysis.score_capture_pair(None, 'a', 'b', None, False)
    assert a is not b and calls == ['a', 'b']


def test_teacher_alignment_guard_is_retained(monkeypatch):
    monkeypatch.setattr(analysis.protocol, 'score_aligned', lambda *a: SimpleNamespace(
        teacher_top1=np.array([1, 2, 3]), realized_token=np.array([4, 4, 4])))
    with pytest.raises(ValueError, match='alignment'):
        analysis.score_capture_pair(None, None, None, None, True)


def test_stale_exact_flag_cannot_reuse_scores_after_capture_changes(monkeypatch):
    monkeypatch.setattr(analysis.protocol, 'VOCAB_LIMIT', 5)
    a = np.arange(15, dtype='<f4').reshape(3, 5)
    b = a.copy()
    metadata = {'original_logit_dtype': 'torch.float32', 'original_logit_width': 5}
    closure = {**analysis.protocol.exact_logits(a, b), **metadata}
    assert analysis.verify_scoring_pair(a, b, metadata, metadata, closure)['exact']
    b[-1, -1] += 1
    with pytest.raises(ValueError, match='closure changed'):
        analysis.verify_scoring_pair(a, b, metadata, metadata, closure)
    with pytest.raises(ValueError, match='representation changed'):
        analysis.verify_scoring_pair(a, a, metadata, {**metadata, 'original_logit_dtype': 'torch.bfloat16'}, closure)


def test_capture_binding_rejects_replaced_metadata_or_stage(tmp_path):
    stage_root = tmp_path / 'full-n128'
    captures = stage_root / 'captures'
    captures.mkdir(parents=True)
    wid = 'conditional-fit-0001'
    meta = captures / f'{wid}.capture.json'
    meta.write_text('original metadata')
    metadata = {'raw_sha256': 'a' * 64}
    receipt = stage_root / 'execution.json'
    receipt.write_text(json.dumps({'windows': [{'id': wid, 'raw_sha256': 'a' * 64,
                                               'capture_sha256': analysis.protocol.sha(meta)}]}))
    execution = {'stages': [{'slot': 'full-n128', 'execution_sha256': analysis.protocol.sha(receipt)}]}
    analysis.bind_capture_to_stage(tmp_path, 'n128', wid, metadata, execution)
    with pytest.raises(ValueError, match='snapshot'):
        analysis.bind_capture_to_stage(tmp_path, 'n128', wid, {'raw_sha256': 'b' * 64}, execution)
    meta.write_text('changed')
    with pytest.raises(ValueError, match='snapshot'):
        analysis.bind_capture_to_stage(tmp_path, 'n128', wid, metadata, execution)
    receipt.write_text('{}')
    with pytest.raises(ValueError, match='stage changed'):
        analysis.bind_capture_to_stage(tmp_path, 'n128', wid, metadata, execution)


def test_json_outputs_never_overwrite_or_serialize_nonfinite_numbers(tmp_path):
    path = tmp_path / 'receipt.json'
    analysis.write_json(path, {'status': 'original'})
    with pytest.raises(FileExistsError):
        analysis.write_json(path, {'status': 'replacement'})
    assert json.loads(path.read_text())['status'] == 'original'
    with pytest.raises(ValueError):
        analysis.write_json(tmp_path / 'bad.json', {'value': float('nan')})
