from glm53_nvfp4.shapley_design import build


def test_antithetic_design_is_complete_and_deterministic():
    design = build(7)
    assert design == build(7)
    assert len(design["permutations"]) == 2
    assert design["permutations"][1]["order"] == list(reversed(design["permutations"][0]["order"]))
    assert all(len(path["coalition_ids"]) == 43 for path in design["permutations"])
    assert design["unique_coalitions"] == 84
