import json
import struct
import subprocess
from pathlib import Path

import pytest

from glm53_nvfp4 import full_coupled_build_support as support


EXPERTS = 2
HIDDEN = 64
INTERMEDIATE = 32  # TP-local
DESIGN_A = "a" * 64
DESIGN_B = "b" * 64
SCALE_SOURCE = "c" * 64


def _tensor_plan(scale_rows: int = 2 * INTERMEDIATE) -> list[tuple[str, str, list[int]]]:
    return [
        ("w13_trellis", "I16", [2, EXPERTS, HIDDEN // 16, INTERMEDIATE // 16, 64]),
        ("w2_trellis", "I16", [EXPERTS, INTERMEDIATE // 16, HIDDEN // 16, 64]),
        ("w13_scale_ue8m0", "U8", [EXPERTS, scale_rows, HIDDEN // 32]),
        ("w2_scale_ue8m0", "U8", [EXPERTS, HIDDEN, INTERMEDIATE // 32]),
        ("gate_up_suh_fp16", "F16", [HIDDEN]),
        ("down_svh_fp16", "F16", [HIDDEN]),
        ("intermediate_scales_fp16", "F16", [EXPERTS, 3 * INTERMEDIATE]),
        ("coupled_sign_draw_u8", "U8", [EXPERTS]),
    ]


_ELEMENT_BYTES = {"I16": 2, "U8": 1, "F16": 2}


def _write_rank(
    path: Path,
    *,
    layer: int,
    rank: int,
    design: str = DESIGN_A,
    fill: int = 0,
    plan: list[tuple[str, str, list[int]]] | None = None,
    metadata_override: dict[str, str] | None = None,
) -> None:
    plan = plan or _tensor_plan()
    header: dict[str, object] = {}
    offset = 0
    for name, dtype, shape in plan:
        count = 1
        for dim in shape:
            count *= dim
        nbytes = count * _ELEMENT_BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    metadata = {
        "schema": support.COUPLED_RANK_SCHEMA,
        "layer": str(layer),
        "rank": str(rank),
        "world_size": "4",
        "bits": "4",
        "alphabet": "e4m3",
        "scale": "ue8m0-k32",
        "law": "procedural-mcg-alpha2",
        "boundary": support.COUPLED_BOUNDARY,
        "full_coupled": "true",
        "ldlq": "false",
        "weight_payload_bpw": "4.25",
        "source_design_sha256": design,
        "exl3_scale_source_sha256": SCALE_SOURCE,
    }
    metadata.update(metadata_override or {})
    header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(bytes([fill]) * offset)


def _write_receipts(receipts: Path, sidecars: Path, layer: int, *, design: str = DESIGN_A, tag: str = "") -> tuple[Path, Path]:
    ranks = []
    for rank in range(4):
        path = support.rank_path(sidecars, layer, rank)
        ranks.append(
            {
                "rank": rank,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": support.sha256_file(path),
                "source_design_sha256": design,
            }
        )
    packer = receipts / f"layer-{layer:03d}{tag}-sidecars.json"
    postwrite = receipts / f"layer-{layer:03d}{tag}-postwrite.json"
    receipts.mkdir(parents=True, exist_ok=True)
    packer.write_text(
        json.dumps(
            {
                "schema": support.COUPLED_SIDECAR_SCHEMA,
                "layer": layer,
                "world_size": 4,
                "boundary": support.COUPLED_BOUNDARY,
                "weight_payload_bpw": 4.25,
                "ldlq": False,
                "ranks": ranks,
            }
        )
    )
    postwrite.write_text(
        json.dumps({"schema": "glm53-p8-coupled-tp4-postwrite-closure.v1", "layer": layer, "status": "pass"})
    )
    return packer, postwrite


def _build_layer(root: Path, layer: int, *, design: str = DESIGN_A) -> tuple[Path, Path]:
    sidecars = root / "sidecars"
    for rank in range(4):
        _write_rank(support.rank_path(sidecars, layer, rank), layer=layer, rank=rank, design=design, fill=layer + rank)
    return _write_receipts(root / "receipts", sidecars, layer, design=design)


def test_inspect_rank_accounts_exact_rate_and_metadata(tmp_path: Path):
    path = support.rank_path(tmp_path, 7, 1)
    _write_rank(path, layer=7, rank=1)
    info = support.inspect_rank(path, layer=7, rank=1, allowed_design_sha256={DESIGN_A}, strict_geometry=False)
    logical = EXPERTS * 3 * HIDDEN * INTERMEDIATE
    assert info["logical_elements"] == logical
    assert info["weight_payload_bytes"] * 8 == logical * 4.25
    assert info["metadata_bytes"] == 2 * HIDDEN * 2 + EXPERTS * 3 * INTERMEDIATE * 2 + EXPERTS
    assert info["bytes"] == path.stat().st_size == info["header_bytes"] + info["weight_payload_bytes"] + info["metadata_bytes"]
    assert info["source_design_sha256"] == DESIGN_A


def test_inspect_rank_rejects_foreign_design_wrong_rate_and_geometry(tmp_path: Path):
    path = support.rank_path(tmp_path, 7, 1)
    _write_rank(path, layer=7, rank=1)
    with pytest.raises(RuntimeError, match="outside the allowlist"):
        support.inspect_rank(path, layer=7, rank=1, allowed_design_sha256={DESIGN_B}, strict_geometry=False)
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        support.inspect_rank(path, layer=8, rank=1, allowed_design_sha256={DESIGN_A}, strict_geometry=False)
    with pytest.raises(RuntimeError, match="shape"):
        support.inspect_rank(path, layer=7, rank=1, allowed_design_sha256={DESIGN_A}, strict_geometry=True)
    # One extra scale row breaks the exact 4.25 bpw relation.
    bad = support.rank_path(tmp_path / "bad", 7, 1)
    _write_rank(bad, layer=7, rank=1, plan=_tensor_plan(scale_rows=2 * INTERMEDIATE + 1))
    with pytest.raises(RuntimeError, match="not exactly 4.25"):
        support.inspect_rank(bad, layer=7, rank=1, allowed_design_sha256={DESIGN_A}, strict_geometry=False)
    identity = support.rank_path(tmp_path / "identity", 7, 1)
    _write_rank(identity, layer=7, rank=1, metadata_override={"full_coupled": "false", "boundary": "identity"})
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        support.inspect_rank(identity, layer=7, rank=1, allowed_design_sha256={DESIGN_A}, strict_geometry=False)


def test_verify_layer_binds_receipts_and_detects_tampering(tmp_path: Path):
    packer, postwrite = _build_layer(tmp_path, 9)
    result = support.verify_layer(
        layer=9, sidecars=tmp_path / "sidecars", packer_receipt=packer, postwrite_receipt=postwrite,
        allowed_design_sha256={DESIGN_A}, strict_geometry=False,
    )
    assert result["status"] == "pass" and len(result["ranks"]) == 4
    assert result["weight_payload_bytes"] * 32 == 17 * result["logical_elements"]
    # Flip one payload byte: sha mismatch must fail closed.
    target = support.rank_path(tmp_path / "sidecars", 9, 2)
    data = bytearray(target.read_bytes()); data[-1] ^= 0xFF; target.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="sha256 differs"):
        support.verify_layer(
            layer=9, sidecars=tmp_path / "sidecars", packer_receipt=packer, postwrite_receipt=postwrite,
            allowed_design_sha256={DESIGN_A}, strict_geometry=False,
        )
    failed = json.loads(postwrite.read_text()); failed["status"] = "fail"; postwrite.write_text(json.dumps(failed))
    with pytest.raises(RuntimeError, match="not a layer-9 pass"):
        support.verify_layer(
            layer=9, sidecars=tmp_path / "sidecars", packer_receipt=packer, postwrite_receipt=postwrite,
            allowed_design_sha256={DESIGN_A}, strict_geometry=False, rehash=False,
        )


def test_reuse_layer_links_and_records_provenance(tmp_path: Path):
    prior = tmp_path / "prior"
    packer, postwrite = _build_layer(prior, 20, design=DESIGN_B)
    # The prior campaign stored each layer in its own subdirectory.
    layer_dir = prior / "sidecars" / "layer-020"
    layer_dir.mkdir()
    for rank in range(4):
        support.rank_path(prior / "sidecars", 20, rank).rename(support.rank_path(layer_dir, 20, rank))
    root = tmp_path / "full"
    args = support.build_parser().parse_args([
        "reuse-layer", "--layer", "20", "--source-dir", str(layer_dir),
        "--source-packer-receipt", str(packer), "--source-postwrite-receipt", str(postwrite),
        "--sidecars", str(root / "sidecars"), "--receipts", str(root / "receipts"),
    ])
    record = support.cmd_reuse_layer(args)
    assert [link["mode"] for link in record["links"]] and all(link["mode"] in {"hardlink", "copy"} for link in record["links"])
    for rank in range(4):
        assert support.rank_path(root / "sidecars", 20, rank).stat().st_size == support.rank_path(layer_dir, 20, rank).stat().st_size
    assert (root / "receipts" / "layer-020-sidecars.json").is_file()
    assert (root / "receipts" / "layer-020-postwrite.json").is_file()
    assert json.loads((root / "receipts" / "layer-020-reuse.json").read_text())["layer"] == 20
    verified = support.verify_layer(
        layer=20, sidecars=root / "sidecars", packer_receipt=root / "receipts" / "layer-020-sidecars.json",
        postwrite_receipt=root / "receipts" / "layer-020-postwrite.json",
        allowed_design_sha256={DESIGN_A, DESIGN_B}, strict_geometry=False,
    )
    assert verified["source_design_sha256"] == DESIGN_B
    with pytest.raises(FileExistsError):
        support.cmd_reuse_layer(args)


def test_manifest_aggregates_all_layers_with_mixed_design_pins(tmp_path: Path):
    root = tmp_path / "full"
    for layer in support.LAYERS:
        _build_layer(root, layer, design=DESIGN_B if layer in (3, 20, 22) else DESIGN_A)
    transform = tmp_path / "transform.json"; transform.write_text("{}\n")
    output = tmp_path / "manifest.json"
    args = support.build_parser().parse_args([
        "manifest", "--sidecars", str(root / "sidecars"), "--receipts", str(root / "receipts"),
        "--output", str(output), "--transform", str(transform),
        "--allow-design-sha256", DESIGN_A, "--allow-design-sha256", DESIGN_B, "--no-strict-geometry",
    ])
    result = support.cmd_manifest(args)
    manifest = json.loads(output.read_text())
    assert manifest["schema"] == support.MANIFEST_SCHEMA
    assert len(manifest["layers"]) == 42
    assert manifest["designs"] == {DESIGN_B: [3, 20, 22], DESIGN_A: [l for l in support.LAYERS if l not in (3, 20, 22)]}
    payload = manifest["payload"]
    assert payload["weight_payload_bpw"] == 4.25
    assert payload["logical_elements"] == 42 * 4 * EXPERTS * 3 * HIDDEN * INTERMEDIATE
    assert payload["file_bytes"] == payload["weight_payload_bytes"] + payload["coupled_metadata_bytes"] + payload["safetensors_header_bytes"]
    assert result["sha256"] == support.sha256_file(output)
    with pytest.raises(FileExistsError):
        support.cmd_manifest(args)
    # A layer outside the allowlist fails the whole manifest.
    (output).unlink()
    args_strict = support.build_parser().parse_args([
        "manifest", "--sidecars", str(root / "sidecars"), "--receipts", str(root / "receipts"),
        "--output", str(output), "--transform", str(transform),
        "--allow-design-sha256", DESIGN_A, "--no-strict-geometry",
    ])
    with pytest.raises(RuntimeError, match="outside the allowlist"):
        support.cmd_manifest(args_strict)


def test_guard_fails_closed_on_production_gpu_and_budget(tmp_path: Path, monkeypatch):
    def fake_run(command, text=True, capture_output=True):
        if command[:1] == ["systemctl"]:
            state = "active" if fake_run.production_active else "inactive"
            return subprocess.CompletedProcess(command, 0, stdout=state + "\n", stderr="")
        if "--query-compute-apps=pid,process_name,used_memory" in command:
            return subprocess.CompletedProcess(command, 0, stdout=fake_run.apps, stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="0, 2, 31\n1, 60, 27\n2, 2, 28\n3, 2, 29\n", stderr="")
    fake_run.production_active = False
    fake_run.apps = ""
    monkeypatch.setattr(support.subprocess, "run", fake_run)
    campaign = tmp_path / "campaign"; campaign.mkdir(); (campaign / "blob").write_bytes(b"x" * 1000)
    base = ["guard", "--root", str(tmp_path), "--campaign-path", str(campaign), "--min-free-bytes", "0"]
    ok = support.build_parser().parse_args(base + ["--max-new-bytes", "5000", "--need-bytes", "1000"])
    assert support.cmd_guard(ok)["status"] == "pass"
    over = support.build_parser().parse_args(base + ["--max-new-bytes", "1500", "--need-bytes", "1000"])
    with pytest.raises(RuntimeError, match="exceed ceiling"):
        support.cmd_guard(over)
    fake_run.apps = "12345, python, 90000 MiB\n"
    with pytest.raises(RuntimeError, match="compute processes"):
        support.cmd_guard(ok)
    fake_run.apps = ""
    fake_run.production_active = True
    with pytest.raises(RuntimeError, match="must remain inactive"):
        support.cmd_guard(ok)


def test_apparent_bytes_charge_hard_links_once_across_campaign_paths(tmp_path: Path):
    a = tmp_path / "k4" / "sidecars"
    b = tmp_path / "mixed" / "sidecars"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "rank-0.safetensors").write_bytes(b"x" * 1000)
    (a / "rank-1.safetensors").write_bytes(b"y" * 500)
    (b / "rank-0.safetensors").hardlink_to(a / "rank-0.safetensors")
    (b / "fresh.safetensors").write_bytes(b"z" * 70)
    # A fresh set per call charges every link; a shared set charges the linked file once.
    assert support.tree_apparent_bytes(tmp_path / "k4") == 1500
    assert support.tree_apparent_bytes(tmp_path / "mixed") == 1070
    charged = support.paths_apparent_bytes([tmp_path / "k4", tmp_path / "mixed"])
    assert charged == {str(tmp_path / "k4"): 1500, str(tmp_path / "mixed"): 70}
    assert sum(charged.values()) == 1570


def test_tree_apparent_bytes_ignores_symlinks(tmp_path: Path):
    (tmp_path / "a").write_bytes(b"1234")
    (tmp_path / "sub").mkdir(); (tmp_path / "sub" / "b").write_bytes(b"12")
    (tmp_path / "link").symlink_to(tmp_path / "a")
    assert support.tree_apparent_bytes(tmp_path) == 6
    assert support.tree_apparent_bytes(tmp_path / "missing") == 0
