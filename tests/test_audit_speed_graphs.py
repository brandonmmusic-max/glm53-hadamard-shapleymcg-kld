import pytest

from glm53_nvfp4.audit_speed_graphs import verify


def logs(label='Capturing CUDA graphs (FULL): 100%', ranks=range(4)):
    return 'enforce_eager=False\nBreakable CUDA graph enabled\n' + label + '\n' + '\n'.join(
        f'(Worker_TP{rank}_DCP{rank}_EP{rank} pid=123) INFO Graph capturing finished in 4 secs'
        for rank in ranks
    )


@pytest.mark.parametrize('label', ['Capturing CUDA graphs (FULL): 100%',
                                 'Capturing decode CUDA graphs (FULL): 100%'])
def test_both_runtime_labels_with_complete_capture(label):
    assert verify(logs(label))['ranks'] == [0, 1, 2, 3]


@pytest.mark.parametrize('label', ['Profiling CUDA graph memory (FULL): 100%',
                                 'Capturing CUDA graphs (FULL): 0%',
                                 'Capturing CUDA graphs (PIECEWISE): 100%'])
def test_incomplete_or_wrong_graph_mode_rejected(label):
    with pytest.raises(ValueError):
        verify(logs(label))


def test_missing_rank_rejected():
    with pytest.raises(ValueError):
        verify(logs(ranks=range(3)))


def test_eager_rejected():
    with pytest.raises(ValueError):
        verify(logs().replace('enforce_eager=False', 'enforce_eager=True'))
