import pytest

from glm53_nvfp4.analyze_p8_smallm_nsys import group_replays, category


def test_correlation_ids_handle_interleaved_replays():
    rows = [{'correlationId':cid,'graphNodeId':node} for cid,node in [(1,9),(2,9),(1,10),(2,10)]]
    groups = group_replays(rows,[1,2])
    assert [[x['graphNodeId'] for x in group] for group in groups] == [[9,10],[9,10]]


def test_refuses_missing_or_unassigned_nodes():
    with pytest.raises(ValueError,match='not assigned'):
        group_replays([{'correlationId':2,'graphNodeId':9}],[1])
    with pytest.raises(ValueError,match='changes'):
        group_replays([{'correlationId':1,'graphNodeId':9},{'correlationId':2,'graphNodeId':10}],[1,2])


def test_fc1_fc2_and_dense_work_remain_separate():
    assert category('foo MoEDynamicKernelBackend_object bar') == 'moe_fc1'
    assert category('foo P8SmallMPhase2Kernel_object bar') == 'moe_fc2'
    assert category('nvjet_sm120_tst_mma') == 'native_dense_mma'
