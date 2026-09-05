import json

from glm53_nvfp4.build_rotation_fit_splits import build


def test_build_rotation_fit_splits_is_balanced_disjoint_and_fit_only(tmp_path):
    rows = []
    for domain_index in range(4):
        domain = f"domain-{domain_index}"
        for index in range(16):
            rows.append(
                {
                    "id": f"{domain}-{index}",
                    "domain": domain,
                    "input_sha256": f"{domain_index:02d}{index:062d}"[-64:],
                    "rank_sha256": f"{index:064x}",
                    "teacher_path": f"fit/{domain}-{index}.safetensors",
                    "token_path": f"tokens/{domain}-{index}.npy",
                    "prediction_positions": 2047,
                }
            )
    source = tmp_path / "roles.json"
    source.write_text(
        json.dumps(
            {
                "source_panel_sha256": "a" * 64,
                "roles": {
                    "fit": rows,
                    "conditional-fit": [],
                    "selection": [],
                    "confirmation": [],
                    "final": [],
                },
            }
        )
    )
    train, tune = build(source)
    assert train["counts"]["fit"] == tune["counts"]["fit"] == 32
    assert train["domains"] == tune["domains"] == {
        f"domain-{index}": 8 for index in range(4)
    }
    train_ids = {row["id"] for row in train["roles"]["fit"]}
    tune_ids = {row["id"] for row in tune["roles"]["fit"]}
    assert not train_ids & tune_ids
    assert all(row["teacher_source_role"] == "fit" for row in train["roles"]["fit"])
    assert not any(train["roles"][role] for role in train["roles"] if role != "fit")
