import pytest

from glm53_nvfp4.analyze_p8_smallm_nsys import (
    group_replays, category, require_c1, select_complete_prefix,
)


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


def test_rejects_non_c1_graph():
    assert require_c1({16}) == 1
    with pytest.raises(ValueError, match='expected active M1'):
        require_c1({32})


def prefix_fixture():
    rows = []
    for cid, offset in ((1, 10), (2, 30)):
        for node in (9, 10):
            rows.append({'correlationId': cid, 'graphNodeId': node,
                         'start': offset + node, 'end': offset + node + 1,
                         'demangledName': node, 'gridX': 16, 'gridY': 1, 'gridZ': 1,
                         'blockX': 256, 'blockY': 1, 'blockZ': 1})
    launches = [{'correlationId': cid, 'start': start, 'end': start + 1, 'globalTid': 7}
                for cid, start in ((1, 11), (2, 31))]
    ranges = [{'start': launch['start'] - 1, 'end': launch['end'] + 1, 'globalTid': 7,
               'eventType': 59} for launch in launches]
    return rows, launches, ranges


def append_tail(rows, launches, ranges):
    rows.append({**rows[0], 'correlationId': 3, 'start': 62, 'end': 63})
    launches.append({'correlationId': 3, 'start': 61, 'end': 64, 'globalTid': 7})
    ranges.append({'start': 60, 'end': 65, 'globalTid': 7, 'eventType': 59})


def test_diagnostic_preserves_partial_tail_and_raw_counts():
    rows, launches, ranges = prefix_fixture()
    append_tail(rows, launches, ranges)
    chunks, exclusions = select_complete_prefix(rows, launches, ranges, 2)
    assert [len(chunk) for chunk in chunks] == [2, 2]
    assert exclusions['raw_graph_launch_count'] == 3
    assert exclusions['raw_generation_range_count'] == 3
    assert exclusions['excluded_graph_kernel_count'] == 1
    assert exclusions['excluded_graph_launches'][0]['correlationId'] == 3
    excluded = exclusions['excluded_generation_ranges'][0]
    assert {key: excluded[key] for key in ranges[-1]} == ranges[-1]
    assert excluded['exclusion_reason'] == 'generation-range-enclosing-trailing-partial-graph'
    # Strict grouping continues to reject the same incomplete tail.
    with pytest.raises(ValueError, match='inventory changes'):
        group_replays(rows, [1, 2, 3])


def test_diagnostic_preserves_negative_duration_nvtx_without_extra_launch():
    rows, launches, ranges = prefix_fixture()
    ranges.append({'start': 1001, 'end': 1000, 'globalTid': 7, 'eventType': 59})
    _, exclusions = select_complete_prefix(rows, launches, ranges, 2)
    assert exclusions['excluded_generation_ranges'][0]['end'] == 1000
    assert exclusions['excluded_graph_launches'] == []


def test_diagnostic_refuses_extra_valid_nvtx_without_matching_tail_launch():
    rows, launches, ranges = prefix_fixture()
    ranges.append({'start': 60, 'end': 65, 'globalTid': 7, 'eventType': 59})
    with pytest.raises(ValueError, match='extra valid generation range'):
        select_complete_prefix(rows, launches, ranges, 2)


@pytest.mark.parametrize('mutation, message', [
    ('early_tail_launch', 'excluded launch'),
    ('early_tail_kernel', 'excluded kernel'),
    ('unknown_tail_node', 'proper subset'),
    ('full_extra_replay', 'proper subset'),
    ('wrong_thread', 'same-thread'),
    ('missing_range', 'same-thread'),
    ('changed_geometry', 'symbol/geometry'),
    ('early_extra_nvtx', 'excluded generation'),
    ('extra_launch', 'at most one tail'),
    ('extra_nvtx', 'at most one excluded'),
])
def test_diagnostic_refuses_unproven_exclusions(mutation, message):
    rows, launches, ranges = prefix_fixture()
    append_tail(rows, launches, ranges)
    if mutation == 'early_tail_launch':
        launches[-1]['start'] = 40
    elif mutation == 'early_tail_kernel':
        rows[-1]['start'] = 40
    elif mutation == 'unknown_tail_node':
        rows[-1]['graphNodeId'] = 99
    elif mutation == 'full_extra_replay':
        rows.append({**rows[1], 'correlationId': 3, 'start': 63, 'end': 64})
    elif mutation == 'wrong_thread':
        ranges[0]['globalTid'] = 8
    elif mutation == 'missing_range':
        ranges.pop(0)
    elif mutation == 'changed_geometry':
        rows[2]['gridX'] = 32
    elif mutation == 'early_extra_nvtx':
        ranges[-1]['start'] = 40
    elif mutation == 'extra_launch':
        launches.append({'correlationId': 4, 'start': 71, 'end': 74, 'globalTid': 7})
    elif mutation == 'extra_nvtx':
        ranges.append({'start': 70, 'end': 75, 'globalTid': 7})
    with pytest.raises(ValueError, match=message):
        select_complete_prefix(rows, launches, ranges, 2)
