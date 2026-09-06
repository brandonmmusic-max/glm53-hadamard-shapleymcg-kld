import json
from pathlib import Path

import pytest

from glm53_nvfp4 import mixed_rate_build_support as mixed
from glm53_nvfp4 import full_coupled_build_support as support
from tests.test_full_coupled_build_support import DESIGN_A, DESIGN_B, EXPERTS, HIDDEN, INTERMEDIATE, _ELEMENT_BYTES, _write_rank  # noqa: F401


def _plan(bits: int):
    return [
        ("w13_trellis", "I16", [2, EXPERTS, HIDDEN // 16, INTERMEDIATE // 16, 16 * bits]),
        ("w2_trellis", "I16", [EXPERTS, INTERMEDIATE // 16, HIDDEN // 16, 16 * bits]),
        ("w13_scale_ue8m0", "U8", [EXPERTS, 2 * INTERMEDIATE, HIDDEN // 32]),
        ("w2_scale_ue8m0", "U8", [EXPERTS, HIDDEN, INTERMEDIATE // 32]),
        ("gate_up_suh_fp16", "F16", [HIDDEN]),
        ("down_svh_fp16", "F16", [HIDDEN]),
        ("intermediate_scales_fp16", "F16", [EXPERTS, 3 * INTERMEDIATE]),
        ("coupled_sign_draw_u8", "U8", [EXPERTS]),
    ]


def _layer(root: Path, layer: int, bits: int, design: str = DESIGN_A):
    sidecars = root / "sidecars"
    for rank in range(4):
        _write_rank(support.rank_path(sidecars, layer, rank), layer=layer, rank=rank, design=design, fill=layer,
                    plan=_plan(bits), metadata_override={"bits": str(bits), "weight_payload_bpw": format(bits + 0.25, ".17g")})
    ranks = []
    for rank in range(4):
        path = support.rank_path(sidecars, layer, rank)
        ranks.append({"rank": rank, "path": str(path), "bytes": path.stat().st_size, "sha256": support.sha256_file(path),
                      "source_design_sha256": design})
    receipts = root / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / f"layer-{layer:03d}-sidecars.json").write_text(json.dumps({
        "schema": support.COUPLED_SIDECAR_SCHEMA, "layer": layer, "world_size": 4, "boundary": support.COUPLED_BOUNDARY,
        "bits": bits, "weight_payload_bpw": bits + 0.25, "ldlq": False, "ranks": ranks}))
    (receipts / f"layer-{layer:03d}-postwrite.json").write_text(json.dumps(
        {"schema": "glm53-p8-coupled-tp4-postwrite-closure.v1", "layer": layer, "status": "pass"}))


def _allocation(path: Path, assignment: dict[int, int]):
    path.write_text(json.dumps({"schema": "glm53.p8-layer-rate-allocation.v1",
                                "assignment": {str(k): v for k, v in assignment.items()}}))


def test_inspect_rank_requires_rate_consistent_metadata_and_payload(tmp_path: Path):
    _layer(tmp_path, 9, 5)
    path = support.rank_path(tmp_path / "sidecars", 9, 0)
    info = mixed.inspect_rank(path, layer=9, rank=0, bits=5, allowed_design_sha256={DESIGN_A})
    assert info["weight_payload_bytes"] * 32 == 21 * info["logical_elements"]
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        mixed.inspect_rank(path, layer=9, rank=0, bits=4, allowed_design_sha256={DESIGN_A})
    with pytest.raises(ValueError):
        mixed.inspect_rank(path, layer=9, rank=0, bits=6, allowed_design_sha256={DESIGN_A})


def test_manifest_installs_allocation_and_enforces_identity_budget(tmp_path: Path):
    assignment = {layer: 4 for layer in support.LAYERS}
    for layer in (5, 6, 7):
        assignment[layer] = 3
    for layer in (30, 31):
        assignment[layer] = 5
    root = tmp_path / "mixed"
    for layer in support.LAYERS:
        _layer(root, layer, assignment[layer], design=DESIGN_B if layer in (3, 20, 22) else DESIGN_A)
    allocation = tmp_path / "allocation.json"
    _allocation(allocation, assignment)
    transform = tmp_path / "transform.json"; transform.write_text("{}")
    output = tmp_path / "manifest.json"
    args = mixed.build_parser().parse_args([
        "manifest", "--allocation", str(allocation), "--sidecars", str(root / "sidecars"), "--receipts", str(root / "receipts"),
        "--output", str(output), "--transform", str(transform), "--allow-design-sha256", DESIGN_A, "--allow-design-sha256", DESIGN_B])
    result = mixed.cmd_manifest(args)
    manifest = json.loads(output.read_text())
    assert result["counts"] == {"3": 3, "4": 37, "5": 2}
    assert manifest["allocation"]["assignment"]["30"] == 5 and manifest["payload"]["same_size_or_smaller"] is True
    assert manifest["payload"]["weight_payload_bpw"] < 4.25
    # Swapping the allocation to disagree with the installed rate must fail closed.
    assignment[30] = 4
    _allocation(allocation, assignment)
    output.unlink()
    with pytest.raises(RuntimeError, match="differs from K4"):
        mixed.cmd_manifest(args)


def test_reuse_layer_links_candidate_sidecars_of_the_allocated_rate(tmp_path: Path):
    source = tmp_path / "candidate"
    _layer(source, 17, 5)
    allocation = tmp_path / "allocation.json"
    _allocation(allocation, {layer: (5 if layer == 17 else 4) for layer in support.LAYERS})
    target = tmp_path / "mixed"
    code = mixed.main(["reuse-layer", "--layer", "17", "--allocation", str(allocation), "--source-dir", str(source / "sidecars"),
                       "--source-packer-receipt", str(source / "receipts" / "layer-017-sidecars.json"),
                       "--source-postwrite-receipt", str(source / "receipts" / "layer-017-postwrite.json"),
                       "--sidecars", str(target / "sidecars"), "--receipts", str(target / "receipts")])
    assert code == 0
    record = json.loads((target / "receipts" / "layer-017-reuse.json").read_text())
    assert record["bits"] == 5 and len(record["links"]) == 4
    verified = mixed.verify_layer(layer=17, bits=5, sidecars=target / "sidecars",
                                  packer_receipt=target / "receipts" / "layer-017-sidecars.json",
                                  postwrite_receipt=target / "receipts" / "layer-017-postwrite.json",
                                  allowed_design_sha256={DESIGN_A})
    assert verified["status"] == "pass"
    # A K3 allocation must refuse the K5 candidate.
    _allocation(allocation, {layer: (3 if layer == 17 else 4) for layer in support.LAYERS})
    code = mixed.main(["reuse-layer", "--layer", "17", "--allocation", str(allocation), "--source-dir", str(source / "sidecars"),
                       "--source-packer-receipt", str(source / "receipts" / "layer-017-sidecars.json"),
                       "--source-postwrite-receipt", str(source / "receipts" / "layer-017-postwrite.json"),
                       "--sidecars", str(tmp_path / "other" / "sidecars"), "--receipts", str(tmp_path / "other" / "receipts")])
    assert code == 1


def test_layer_complete_cli_reads_rate_from_allocation(tmp_path: Path):
    _layer(tmp_path, 12, 3)
    allocation = tmp_path / "allocation.json"
    _allocation(allocation, {layer: (3 if layer == 12 else 4) for layer in support.LAYERS})
    code = mixed.main(["layer-complete", "--layer", "12", "--allocation", str(allocation), "--sidecars", str(tmp_path / "sidecars"),
                       "--packer-receipt", str(tmp_path / "receipts" / "layer-012-sidecars.json"),
                       "--postwrite-receipt", str(tmp_path / "receipts" / "layer-012-postwrite.json"),
                       "--allow-design-sha256", DESIGN_A])
    assert code == 0
