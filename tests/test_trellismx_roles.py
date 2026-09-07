import json
from pathlib import Path

import pytest

from trellismx.roles import (
    QAD_FUTURE_REFERENCE,
    RoleManifest,
    evaluation_reference,
)


def test_loads_legacy_sealed_roles_and_keeps_them_disjoint() -> None:
    path = Path(__file__).resolve().parents[1] / "evidence/sealed/v4/roles-v4.json"
    manifest = RoleManifest.from_file(path)
    summary = manifest.summary()
    assert manifest.schema == "glm53-nvfp4-v4.roles.v1"
    assert summary["counts"] == {
        "fit": 64,
        "conditional-fit": 0,
        "selection": 32,
        "confirmation": 28,
        "final": 0,
    }
    assert summary["qad_used_for_this_manifest"] is False


def test_rejects_same_input_in_multiple_roles() -> None:
    digest = "a" * 64
    value = {
        "schema": "trellismx.roles.v1",
        "roles": {
            "fit": [
                {"id": "one", "input_sha256": digest, "prediction_positions": 10},
                {"id": "two", "input_sha256": digest, "prediction_positions": 10},
            ],
            "conditional-fit": [],
            "selection": [],
            "confirmation": [],
            "final": [],
        },
    }
    with pytest.raises(ValueError, match="cannot occur in multiple roles"):
        RoleManifest.from_mapping(value)


def test_qad_is_future_reference_not_prior_measurement_provenance() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/glm53-flash-qad-future-evaluation-v1.json"
    )
    value = evaluation_reference(json.loads(path.read_text()))
    assert value == QAD_FUTURE_REFERENCE
    assert value["used_for_existing_trellismx_measurements"] is False
    assert value["partitions"]["qualification"]["scored_positions"] == 524_020
