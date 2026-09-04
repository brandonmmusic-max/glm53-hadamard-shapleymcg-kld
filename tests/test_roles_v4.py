import hashlib
import json

import numpy as np

from glm53_nvfp4.roles_v4 import DOMAINS, build


def _window(arrays, role, domain, index, ordinal):
    wid = f"{role}-{domain}-{index:04d}"
    tokens = np.arange(2048, dtype=np.int32) + ordinal * 4096
    path = arrays / f"{wid}.tokens.npy"
    np.save(path, tokens)
    return {
        "window_id": wid,
        "role": role,
        "domain": domain,
        "prediction_positions": 2047,
        "token_ids_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_v4_uses_only_never_declared_conditional_rows_for_selection(tmp_path):
    arrays = tmp_path / "arrays"
    arrays.mkdir()
    windows = []
    ordinal = 0
    for role, count in (("fit", 16), ("conditional-fit", 16), ("confirmation", 8)):
        for domain in DOMAINS:
            for index in range(count):
                windows.append(_window(arrays, role, domain, index, ordinal))
                ordinal += 1
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({"windows": windows}))

    by = {item["window_id"]: item for item in windows}
    fit = [item for item in windows if item["role"] == "fit"]
    v2_used = [
        item for item in windows
        if item["role"] == "conditional-fit" and int(item["window_id"].rsplit("-", 1)[1]) < 4
    ]
    v3_used = [
        item for item in windows
        if item["role"] == "conditional-fit" and 4 <= int(item["window_id"].rsplit("-", 1)[1]) < 8
    ]
    confirmation = [item for item in windows if item["role"] == "confirmation"][:28]

    def entry(item):
        path = arrays / f"{item['window_id']}.tokens.npy"
        return {
            "id": item["window_id"],
            "domain": item["domain"],
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "token_path": str(path),
        }

    v2 = tmp_path / "roles-v2.json"
    v2.write_text(json.dumps({"roles": {"fit": [entry(x) for x in fit], "conditional-fit": [entry(x) for x in v2_used], "selection": [], "confirmation": [], "final": []}}))
    v3 = tmp_path / "roles-v3.json"
    v3.write_text(json.dumps({"roles": {"fit": [entry(x) for x in fit], "conditional-fit": [entry(x) for x in v3_used], "selection": [], "confirmation": [entry(x) for x in confirmation], "final": []}}))

    result = build(panel, arrays, v2, v3)
    assert result["counts"] == {
        "fit": 64,
        "conditional-fit": 0,
        "selection": 32,
        "confirmation": 28,
        "final": 0,
    }
    selected = result["roles"]["selection"]
    assert {item["teacher_source_role"] for item in selected} == {"conditional-fit"}
    assert {item["experimental_role"] for item in selected} == {"selection"}
    used = {item["window_id"] for item in v2_used + v3_used}
    assert not ({item["id"] for item in selected} & used)
    assert result == build(panel, arrays, v2, v3)
