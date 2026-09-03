import hashlib
import json

import numpy as np

from glm53_nvfp4.roles import build


def test_role_builder_is_deterministic_and_disjoint(tmp_path):
    arrays = tmp_path / "arrays"
    arrays.mkdir()
    windows = []
    domains = ["a", "b", "c", "d"]
    roles = {"fit": 16, "conditional-fit": 4, "selection": 4, "confirmation": 8}
    for role, count in roles.items():
        for domain_index, domain in enumerate(domains):
            for index in range(count + 1):
                wid = f"{role}-{domain}-{index:04d}"
                tokens = np.arange(2048, dtype=np.int32) + len(windows) * 4096
                np.save(arrays / f"{wid}.tokens.npy", tokens)
                windows.append({
                    "window_id": wid,
                    "role": role,
                    "domain": domain,
                    "prediction_positions": 2047,
                    "token_ids_sha256": hashlib.sha256((arrays / f"{wid}.tokens.npy").read_bytes()).hexdigest(),
                })
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({"sealed_corpus_sha256": "0" * 64, "windows": windows}))
    first = build(panel, arrays)
    second = build(panel, arrays)
    assert first == second
    assert first["counts"] == {"conditional-fit": 16, "confirmation": 32, "final": 0, "fit": 64, "selection": 16}
    hashes = [entry["input_sha256"] for items in first["roles"].values() for entry in items]
    assert len(hashes) == len(set(hashes)) == 128
