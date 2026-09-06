import copy
from test_p8_coupled_graph_device_closure import _module


def test_only_repeat_and_physical_permutation_are_ignored():
    module = _module()
    a = dict(repeat=0, hashes=dict(output='same'),schedule=dict(route_to_physical_sha256='a',routes=512))
    b = copy.deepcopy(a)
    b['repeat'] = 1
    b['schedule']['route_to_physical_sha256'] = 'b'
    assert module._replay_comparison(a) == module._replay_comparison(b)
    assert a['schedule']['route_to_physical_sha256'] == 'a'
    b['hashes']['output'] = 'corrupt'
    assert module._replay_comparison(a) != module._replay_comparison(b)
    b['hashes']['output'] = 'same'
    b['schedule']['routes'] = 511
    assert module._replay_comparison(a) != module._replay_comparison(b)
