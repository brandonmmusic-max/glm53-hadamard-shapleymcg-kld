import torch

from glm53_nvfp4.analyze_p8_joint_route_policy import coordinate_policy
from glm53_nvfp4.analyze_p8_joint_route_policy_v2 import coordinate_policy_three


def test_coordinate_policy_accounts_for_corouted_cancellation():
    identity = torch.tensor([[2.0, 0.0]])
    rows = [torch.tensor([0]), torch.tensor([0])]
    deltas = [torch.tensor([[-1.0, 0.0]]), torch.tensor([[-1.0, 0.0]])]
    starts = [torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool)]
    mask, sse, trace = coordinate_policy(identity, rows, deltas, starts, 4)
    assert mask.tolist() == [True, True]
    assert sse == 0.0
    assert len(trace) == 2


def test_coordinate_policy_can_reject_individually_harmful_delta():
    identity = torch.tensor([[0.25]])
    rows = [torch.tensor([0])]
    deltas = [torch.tensor([[1.0]])]
    starts = [torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool)]
    mask, sse, _ = coordinate_policy(identity, rows, deltas, starts, 4)
    assert mask.tolist() == [False]
    assert sse == 0.0625


def test_three_way_policy_selects_fc1_only_state():
    identity = torch.tensor([[2.0]])
    rows = [torch.tensor([0])]
    full = [torch.tensor([[1.0]])]
    fc1 = [torch.tensor([[-2.0]])]
    starts = [torch.zeros(1, dtype=torch.uint8)]
    states, sse, _ = coordinate_policy_three(identity, rows, full, fc1, starts, 4)
    assert states.tolist() == [2]
    assert sse == 0.0
