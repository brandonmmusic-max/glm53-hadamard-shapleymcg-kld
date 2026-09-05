"""Independent negative fixtures for forced-M1 analysis receipts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest

from glm53_nvfp4 import p8_decode_analysis as analysis


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


@pytest.fixture
def receipt_tree(tmp_path, monkeypatch):
    root = tmp_path / "capture"
    root.mkdir()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    image = "sha256:" + "a" * 64
    window = {
        "id": "conditional-fit-0001",
        "domain": "axis1_general",
        "token_path": str(tmp_path / "tokens.npy"),
    }
    order = [
        {"stage": stage, "arm": arm}
        for stage in ("canary", "full")
        for arm in ("n128", "n64")
    ]
    monkeypatch.setattr(analysis.launcher, "ORDER", order)
    monkeypatch.setattr(
        analysis.launcher, "stage_windows", lambda plan, stage: [window]
    )
    monkeypatch.setattr(
        analysis.launcher, "runtime_audit", lambda log, arm, windows: {}
    )
    monkeypatch.setattr(
        analysis.launcher,
        "compare_stage",
        lambda plan, output, stage: {
            "stage": stage,
            "all_exact": True,
            "windows": [{"window_id": window["id"], "exact": True}],
        },
    )
    monkeypatch.setattr(analysis.protocol, "completion_request", lambda *args: {})
    monkeypatch.setattr(analysis.protocol, "verify_response", lambda *args: {})
    monkeypatch.setattr(analysis.np, "load", lambda *args, **kwargs: object())

    stages = {}
    top_stages = []
    plan_hash = analysis.protocol.sha(plan_path)
    base = datetime(2026, 9, 5, tzinfo=timezone.utc)
    for index, entry in enumerate(order):
        slot = analysis.launcher.slot_name(entry)
        folder = root / slot
        begin = base + timedelta(seconds=20 * index)
        end = begin + timedelta(seconds=10)
        cid = f"{index + 1:064x}"
        files = {
            "cooldown.jsonl": json.dumps({"temperatures": [60, 61, 62, 63]})
            + "\n",
            "thermal.jsonl": json.dumps({"temperatures": [70, 71, 72, 73]})
            + "\n",
            "server-final.private.log": "runtime proof\n",
            "container-final.private.json": json.dumps(
                {
                    "Id": cid,
                    "Image": image,
                    "Name": f"/{analysis.launcher.PREFIX}-{slot}",
                    "Created": (begin + timedelta(seconds=1))
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "State": {
                        "Running": False,
                        "StartedAt": (begin + timedelta(seconds=2))
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "FinishedAt": (begin + timedelta(seconds=9))
                        .isoformat()
                        .replace("+00:00", "Z"),
                    },
                    "Config": {
                        "Labels": {
                            analysis.launcher.cold.LABEL: f"{plan_hash}:{slot}"
                        }
                    },
                }
            ),
            "runtime-final-audit.json": "{}",
            f"requests/{window['id']}.request.json": "{}",
            f"requests/{window['id']}.response.json": "{}",
            f"captures/{window['id']}.capture.json": "{}",
            f"captures/{window['id']}.logits.f32": "raw",
        }
        for relative, content in files.items():
            _write(folder / relative, content)
        stage = {
            **entry,
            "schema": "glm53-p8.forced-m1-v2-stage.v1",
            "started_at": begin.isoformat(),
            "finished_at": end.isoformat(),
            "plan_sha256": plan_hash,
            "capture_image": image,
            "exit_code": 0,
            "cleanup": {"ok": True, "errors": []},
            "unreadable_artifacts": [],
            "protected_roles_opened": [],
            "allocation_restart": False,
            "container_id": cid,
            "windows": [{"id": window["id"]}],
            "files": {
                relative: {
                    "bytes": (folder / relative).stat().st_size,
                    "sha256": analysis.protocol.sha(folder / relative),
                }
                for relative in files
            },
        }
        _write(folder / "execution.json", stage)
        stages[slot] = stage
        top_stages.append(
            {
                "slot": slot,
                "execution_sha256": analysis.protocol.sha(folder / "execution.json"),
            }
        )

    for stage in ("canary", "full"):
        _write(
            root / f"{stage}-exact.json",
            {
                "stage": stage,
                "all_exact": True,
                "windows": [{"window_id": window["id"], "exact": True}],
            },
        )
    execution = {
        "schema": "glm53-p8.forced-m1-v2-execution.v1",
        "plan_sha256": plan_hash,
        "capture_image": image,
        "exit_code": 0,
        "capture_protocol_complete": True,
        "allocation_restart": False,
        "opened_roles": ["conditional-fit"],
        "protected_roles_opened": [],
        "restoration_safety": {"ok": True},
        "prior": {"backend": True, "timer": False},
        "restoration": {"backend": True, "timer": False, "errors": []},
        "stages": top_stages,
        "full_exact": True,
    }

    def reseal(slot=None):
        if slot is not None:
            folder = root / slot
            _write(folder / "execution.json", stages[slot])
            next(row for row in top_stages if row["slot"] == slot)[
                "execution_sha256"
            ] = analysis.protocol.sha(folder / "execution.json")
        _write(root / "execution.json", execution)

    reseal()
    plan = {"output": str(root), "capture_image": image, "windows": [window]}
    return plan, plan_path, root, stages, execution, reseal


def test_valid_execution_receipt_fixture_replays(receipt_tree):
    plan, plan_path, _, _, _, _ = receipt_tree
    _, closure = analysis.verify_execution(plan, plan_path)
    assert closure["all_exact"] is True


@pytest.mark.parametrize("mutation,match", [
    ("role", "incomplete or unauthenticated"),
    ("mandatory", "mandatory capture files"),
    ("cooldown", "thermal/start gate"),
])
def test_execution_receipt_role_omission_and_nonfinite_cooling_fail_closed(
    receipt_tree, mutation, match
):
    plan, plan_path, root, stages, execution, reseal = receipt_tree
    if mutation == "role":
        execution["opened_roles"] = ["selection"]
        reseal()
    elif mutation == "mandatory":
        slot = "full-n64"
        stages[slot]["files"].pop(f"requests/{plan['windows'][0]['id']}.response.json")
        reseal(slot)
    else:
        slot = "canary-n128"
        path = root / slot / "cooldown.jsonl"
        _write(path, '{"temperatures":[NaN,60,60,60]}\n')
        stages[slot]["files"]["cooldown.jsonl"] = {
            "bytes": path.stat().st_size,
            "sha256": analysis.protocol.sha(path),
        }
        reseal(slot)
    with pytest.raises(ValueError, match=match):
        analysis.verify_execution(plan, plan_path)
