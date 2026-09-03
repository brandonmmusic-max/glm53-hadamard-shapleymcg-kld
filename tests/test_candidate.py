import json

import torch
from safetensors.torch import save_file

from glm53_nvfp4.candidate import build


def test_candidate_redirects_only_chunk_tensors(tmp_path):
    carrier = tmp_path / "carrier"
    carrier.mkdir()
    save_file({"a.weight": torch.ones(2, 2), "b.weight": torch.zeros(2, 2)}, str(carrier / "base.safetensors"))
    (carrier / "config.json").write_text("{}")
    (carrier / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {"a.weight": "base.safetensors", "b.weight": "base.safetensors"}}))
    chunk = tmp_path / "new.safetensors"
    save_file({"a.weight": torch.full((2, 2), 3.0)}, str(chunk))
    output = tmp_path / "candidate"
    result = build(carrier, output, [chunk])
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert result["redirected_tensors"] == 1
    assert index["weight_map"]["a.weight"] == "new.safetensors"
    assert index["weight_map"]["b.weight"] == "base.safetensors"
    assert (output / "base.safetensors").is_symlink()
