import json

from glm53_nvfp4.finalize_selection_stop import update_record


def test_selection_stop_completes_record_without_opening_confirmation(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}")
    final = tmp_path / "final.json"
    final.write_text("{}")
    terminal = tmp_path / "terminal.json"
    terminal.write_text("{}")
    record = {
        "status": "executing",
        "authorization": {"compute": "authorized"},
        "deviations": [],
        "invariants": [
            {"name": name, "observed": "Not tested", "status": "pending"}
            for name in ("source_integrity", "pilot_export_load", "candidate_freeze", "restoration")
        ],
    }
    selection = {
        "candidate_mean_kld": 0.11,
        "stock_mean_kld": 0.10,
        "mean_delta_kld": 0.01,
        "relative_improvement": -0.10,
        "delta_ci95_percentile": [-0.01, 0.03],
        "delta_ci95_bca": [-0.01, 0.03],
    }
    update_record(record, selection=selection, terminal_receipt=terminal, source_verify=source, final_state=final)
    assert record["status"] == "completed"
    assert record["qualification"]["decision"] == "selection-stop"
    assert record["qualification"]["attempt_count"] == 0
    assert record["authorization"]["compute"] == "completed"
    freeze = next(x for x in record["invariants"] if x["name"] == "candidate_freeze")
    assert freeze["status"] == "pending"
    assert {item["stage"] for item in record["deviations"]} == {
        "source-integrity-timing",
        "terminal-inventory-validation",
    }
