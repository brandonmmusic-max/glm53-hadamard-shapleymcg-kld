from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest
from safetensors.numpy import load
import torch

from glm53_nvfp4 import p4_encoder
from glm53_nvfp4.block_gptq import _prepare_full_inverse
from glm53_nvfp4.p4_codec import (
    decode_p4, e2m1_codes, e2m1_values, fit_e4m3_scales,
    mcg_half_values, reconstruct_k4_states, serialize_p4, unpack_e2m1, unpack_k4_edges,
)
from glm53_nvfp4.p4_fixture import scalar_mcg_code
from glm53_nvfp4.p4_serving_codec import (
    LAW, PROJECTIONS, RANK_CONTRACT, assemble_rank, deserialize_rank, export_tp4,
    rank_accounting, rank_metadata, rank_projection, read_rank, serialize_rank,
    validate_rank,
)
from glm53_nvfp4.p4_serving_fixture import (
    DESIGN_SHA, fixture_expectations, make_experts, matrix_values,
    quantize_activations, write_matrix_fixture,
)


@pytest.fixture(scope="module")
def experts():
    return make_experts()


def _rank(experts, rank=0):
    return assemble_rank(experts, layer=3, rank=rank, source_design_sha256=DESIGN_SHA)


def edit_header(raw, fn):
    size = struct.unpack_from("<Q", raw)[0]
    head = json.loads(raw[8:8 + size])
    fn(head)
    encoded = json.dumps(head, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + raw[8 + size:]


def test_encoder_lut_all_65536_native_rne_states_and_signed_zero():
    lut = p4_encoder.encoder_state_lut()
    assert lut.dtype == torch.uint8 and lut.numel() == 65536
    values = lut.view(torch.float8_e4m3fn).float().numpy()
    native = e2m1_codes(values)
    expected = np.array([scalar_mcg_code(state) for state in range(65536)], dtype=np.uint8)
    np.testing.assert_array_equal(native, expected)
    np.testing.assert_array_equal(np.signbit(values), (expected & 8) != 0)
    assert np.any(native == 8) and np.any(native == 0)
    assert p4_encoder.encoder_lut_sha256() == hashlib.sha256(lut.numpy().tobytes()).hexdigest()
    # Old paths must be re-encoded; the mismatch is not merely a schema label.
    from glm53_nvfp4.trellis_nvfp4 import e2m1_state_lut
    assert not torch.equal(lut, e2m1_state_lut(4, law="mcg", compander_scale=1))


def test_native_selection_adapter_propagates_exact_lut_and_rejects_backend_drift(monkeypatch):
    lut = p4_encoder.encoder_state_lut()
    source = torch.zeros(1, 256)
    seen = []

    def backend(tiles, state_lut, *, bits, tailbite_context):
        seen.append((state_lut.clone(), bits, tailbite_context))
        states = torch.full_like(tiles, 0x7777, dtype=torch.int32)
        return state_lut.view(torch.float8_e4m3fn).float()[states.long()], states

    monkeypatch.setattr(p4_encoder, "_encode_tiles", backend)
    values, states = p4_encoder.checked_select_tiles(source, lut, tailbite_context=128)
    assert torch.equal(seen[0][0], lut) and seen[0][1:] == (4, 128)
    assert torch.all(values == .5) and torch.all(states == 0x7777)
    changed = lut.clone()
    changed[0] ^= 1
    with pytest.raises(ValueError, match="frozen"):
        p4_encoder.checked_select_tiles(source, changed, tailbite_context=128)
    monkeypatch.setattr(p4_encoder, "_encode_tiles", lambda *a, **kw: (torch.zeros_like(source), torch.zeros_like(source, dtype=torch.int32)))
    with pytest.raises(ValueError, match="labels disagree"):
        p4_encoder.checked_select_tiles(source, lut, tailbite_context=128)


def test_calibration_hessian_changes_later_group_selection_and_inverse_reuse(monkeypatch):
    """Structural sensitivity: a finite candidate-path selector stands in for CUDA Viterbi.

    This exercises the actual full-Hessian GPTQ feedback function. It does not
    claim this tiny selector is CUDA Viterbi or prove GLM quality.
    """
    lut = p4_encoder.encoder_state_lut()
    candidate_states = torch.arange(16, dtype=torch.int64) * 0x1111
    labels = lut.view(torch.float8_e4m3fn).float()[candidate_states]
    calls = []

    def candidate_path_selector(tiles, state_lut, *, bits, tailbite_context):
        assert torch.equal(state_lut, lut) and bits == 4
        calls.append(tiles.clone())
        selected = ((tiles[:, None, :] - labels[None, :, None]) ** 2).sum(-1).argmin(-1)
        states = candidate_states[selected, None].expand_as(tiles).clone()
        return labels[selected, None].expand_as(tiles).clone(), states

    monkeypatch.setattr(p4_encoder, "_encode_tiles", candidate_path_selector)
    weight = torch.cat((torch.full((16, 16), .37), torch.full((16, 16), .79)), dim=1)
    diagonal = torch.eye(32)
    correlated = diagonal.clone()
    correlated[:16, 16:] = .8 * torch.eye(16)
    correlated[16:, :16] = .8 * torch.eye(16)
    scales, gs = torch.ones(16, 2).to(torch.float8_e4m3fn), torch.tensor(1.)
    serial, serial_states = p4_encoder.feedback_select(weight, diagonal, scales, gs, lut, column_block=32)
    first_inputs = [item.clone() for item in calls]
    calls.clear()
    aware, aware_states = p4_encoder.feedback_select(weight, correlated, scales, gs, lut, column_block=32)
    assert torch.equal(first_inputs[0], calls[0])
    assert not torch.equal(first_inputs[1], calls[1])
    assert not torch.equal(serial_states[1], aware_states[1])
    assert torch.all(serial[:, 16:] == 1.) and torch.all(aware[:, 16:] == .5)
    prepared = _prepare_full_inverse(correlated, .01, 16)
    cached, cached_states = p4_encoder.feedback_select(weight, correlated, scales, gs, lut,
        column_block=32, prepared_inverse=prepared)
    assert torch.equal(cached, aware) and torch.equal(cached_states, aware_states)
    # Within-group act order cannot change the stored K16 group membership.
    assert torch.equal(prepared[1] // 16, torch.arange(32) // 16)


def test_positive_scale_refit_and_activation_policy_match_independent_oracles():
    rng = np.random.default_rng(21)
    codes = rng.integers(0, 16, (16, 32), dtype=np.uint8)
    codes[0, :16] = 8
    source = rng.normal(size=(16, 32)).astype("<f4")
    basis = e2m1_values(codes)
    gs = np.array(.25, dtype="<f4")
    expected = fit_e4m3_scales(source, codes, gs)
    result = p4_encoder.refit_scales(torch.from_numpy(source), torch.from_numpy(basis).reshape(16, 2, 16), torch.from_numpy(gs))
    np.testing.assert_array_equal(result.view(torch.uint8).numpy(), expected)
    source[1] = 0
    actual = p4_encoder.qdq_p4_activations(torch.from_numpy(source)).double().numpy()
    np.testing.assert_array_equal(actual, quantize_activations(source.astype(np.float64)))


def test_every_rank_reconstructs_exact_original_matrix_bytes_and_globals(experts):
    per_rank = [_rank(experts, rank) for rank in range(4)]
    for expert, projections in enumerate(experts):
        for projection, payload in projections.items():
            original = decode_p4(payload)
            pieces = [decode_p4(rank_projection(tensors, metadata, expert=expert, projection=projection))
                      for tensors, metadata in per_rank]
            axis = 1 if projection == "down_proj" else 0
            np.testing.assert_array_equal(np.concatenate([piece.weight for piece in pieces], axis=axis), original.weight)
            np.testing.assert_array_equal(np.concatenate([piece.scale_e4m3 for piece in pieces], axis=axis), original.scale_e4m3)
            assert all(piece.global_scale.tobytes() == original.global_scale.tobytes() for piece in pieces)
            vectorized = p4_encoder.payload_reconstruction(payload).numpy()
            np.testing.assert_array_equal(vectorized, matrix_values(payload).astype(np.float32))


def test_encoder_state_export_preserves_signed_carrier_and_negative_zero(experts):
    original = experts[0]["gate_proj"]
    states = reconstruct_k4_states(unpack_k4_edges(original.trellis))
    signed = torch.from_numpy(states.view("<i2").copy())
    scales = torch.from_numpy(original.scale_e4m3).view(torch.float8_e4m3fn)
    payload = p4_encoder.payload_from_states(signed, scales, torch.from_numpy(original.global_scale),
        rows=original.rows, width=original.width)
    assert serialize_p4(payload) == serialize_p4(original)
    nibble = unpack_e2m1(decode_p4(payload).weight)
    assert np.any(nibble == 8)
    bad = signed.clone()
    bad[0, 0, 0] ^= 0x10
    with pytest.raises(ValueError, match="recurrence"):
        p4_encoder.payload_from_states(bad, scales, torch.from_numpy(original.global_scale),
            rows=original.rows, width=original.width)


def test_full_fc1_fc2_cpu_fixture_tp4_sum(experts):
    result = fixture_expectations(experts)
    assert result["tp4_sum_max_abs"] == 0
    assert result["gpu_execution"] == result["kld"] == result["speed"] == "Not tested"
    assert set(result["stage_sha256"]) == {"gate_up", "middle", "down"}


def test_ordinary_and_strict_safetensors_roundtrip_and_exact_bpw(experts):
    tensors, metadata = _rank(experts, 2)
    raw = serialize_rank(tensors, metadata)
    ordinary = load(raw)
    restored, meta = deserialize_rank(raw, expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_layer=3, expected_rank=2, expected_design_sha256=DESIGN_SHA, expected_experts=3)
    for name in tensors:
        assert ordinary[name].tobytes() == restored[name].tobytes() == tensors[name].tobytes()
    assert serialize_rank(restored, meta) == raw
    rates = rank_accounting(tensors, metadata)
    assert rates["global_scale_bytes"] == 3 * 3 * 4
    assert rates["stream_and_scales_bpw"] == {"numerator": 9, "denominator": 2, "decimal": 4.5}
    expected = Fraction(9, 2) + Fraction(32, 128 * 64)
    assert Fraction(rates["payload_bpw"]["numerator"], rates["payload_bpw"]["denominator"]) == expected
    assert rates["file_bytes"] == len(raw)
    assert rates["payload_bytes"] == sum(t.nbytes for t in tensors.values())
    assert rates["runtime_state_lut_bytes"] == 0
    assert set(ordinary) == set(tensors) and len(tensors) == 6


@pytest.mark.parametrize("key,value", [("law", "procedural-mcg"), ("weight_rounding", "nearest-ties-low-magnitude"),
    ("signed_zero", "canonical-positive"), ("scale", "ue8m0-k32"), ("w13_order", "up,gate"),
    ("ldlq", "true"), ("world_size", "2"), ("intermediate_per_rank", "256")])
def test_reject_legacy_and_wrong_contract_metadata(experts, key, value):
    tensors, metadata = _rank(experts)
    metadata[key] = value
    with pytest.raises(ValueError):
        validate_rank(tensors, metadata)


@pytest.mark.parametrize("bad", [0, 127, 128, 255])
def test_reject_invalid_weight_scale_bytes(experts, bad):
    tensors, metadata = _rank(experts)
    tensors["w13_scale_e4m3"][0, 0, 0, 0] = bad
    with pytest.raises(ValueError, match="positive"):
        serialize_rank(tensors, metadata)


@pytest.mark.parametrize("bad", [0., -1., float("inf"), float("nan")])
def test_reject_invalid_globals(experts, bad):
    tensors, metadata = _rank(experts)
    tensors["w2_global_scale"][0] = bad
    with pytest.raises(ValueError, match="positive finite"):
        serialize_rank(tensors, metadata)


def test_reject_extra_missing_malformed_or_corrupt_tensors(experts):
    tensors, metadata = _rank(experts)
    with pytest.raises(ValueError, match="six"):
        serialize_rank({**tensors, "hidden_codebook": np.zeros(65536, dtype=np.uint8)}, metadata)
    with pytest.raises(ValueError, match="six"):
        serialize_rank({name: value for name, value in tensors.items() if name != "w2_global_scale"}, metadata)
    raw = serialize_rank(tensors, metadata)
    bad_raw = raw[:-1] + bytes([raw[-1] ^ 1])
    with pytest.raises(ValueError, match="hash"):
        deserialize_rank(bad_raw)
    for bad in (raw[:-1], raw + b"x", edit_header(raw, lambda h: h["w2_trellis"]["shape"].reverse()),
                edit_header(raw, lambda h: h["w13_global_scale"].update(dtype="F16"))):
        with pytest.raises(ValueError):
            deserialize_rank(bad)
    with pytest.raises(ValueError, match="rank mismatch"):
        deserialize_rank(raw, expected_rank=3)
    with pytest.raises(ValueError, match="design mismatch"):
        deserialize_rank(raw, expected_design_sha256="f" * 64)


def test_export_reads_each_matrix_once_matches_serial_and_resumes_without_rewrite(tmp_path, experts):
    source = write_matrix_fixture(tmp_path / "source", experts)
    target = tmp_path / "ranks"
    result = export_tp4(source, target, expected_design_sha256=DESIGN_SHA)
    mtimes = {}
    for entry in result["entries"]:
        path = target / entry["path"]
        serial, metadata = _rank(experts, entry["rank"])
        assert path.read_bytes() == serialize_rank(serial, metadata)
        mtimes[path] = path.stat().st_mtime_ns
    timings = list(target.glob("export-timings-*.json"))
    assert len(timings) == 1
    first = json.loads(timings[0].read_text())
    assert first["source_matrix_reads"] == 9 and first["written_ranks"] == 4
    repeat = export_tp4(source, target, expected_design_sha256=DESIGN_SHA, resume=True)
    assert repeat == result
    assert all(path.stat().st_mtime_ns == value for path, value in mtimes.items())
    second = next(json.loads(path.read_text()) for path in target.glob("export-timings-*.json") if path != timings[0])
    assert second["source_matrix_reads"] == 0 and second["reused_ranks"] == 4
    assert result["total"]["global_scale_bytes_including_tp_replication"] == 4 * 3 * 3 * 4
    with pytest.raises(FileExistsError):
        export_tp4(source, target, expected_design_sha256=DESIGN_SHA)


def test_export_resumes_completed_prefix_and_keeps_failed_partial(tmp_path, experts, monkeypatch):
    from glm53_nvfp4 import p4_serving_codec as bridge
    source = write_matrix_fixture(tmp_path / "source", experts)
    target = tmp_path / "ranks"
    original = bridge.serialize_rank
    counter = []
    def interrupted(tensors, metadata):
        counter.append(metadata["rank"])
        if metadata["rank"] == "1":
            raise RuntimeError("deliberate interrupted export fixture")
        return original(tensors, metadata)
    monkeypatch.setattr(bridge, "serialize_rank", interrupted)
    with pytest.raises(RuntimeError, match="deliberate"):
        export_tp4(source, target, expected_design_sha256=DESIGN_SHA)
    rank0 = target / "p4-layer-003-tp4-rank-0.safetensors"
    before = rank0.stat().st_mtime_ns
    assert not (target / "manifest.json").exists()
    monkeypatch.setattr(bridge, "serialize_rank", original)
    result = export_tp4(source, target, expected_design_sha256=DESIGN_SHA, resume=True)
    assert rank0.stat().st_mtime_ns == before and len(result["entries"]) == 4
    for entry in result["entries"]:
        assert (target / entry["path"]).read_bytes() == original(*_rank(experts, entry["rank"]))


def test_export_rejects_incomplete_mixed_and_untrusted_source_manifests(tmp_path, experts):
    source = write_matrix_fixture(tmp_path / "source", experts)
    original = json.loads(source.read_text())
    def save(data):
        source.write_text(json.dumps(data))
    incomplete = {**original, "matrices": original["matrices"][:-1]}
    save(incomplete)
    with pytest.raises(ValueError, match="missing"):
        export_tp4(source, tmp_path / "out", expected_design_sha256=DESIGN_SHA)
    save({**original, "matrices": original["matrices"] + [original["matrices"][0]]})
    with pytest.raises(ValueError, match="duplicate"):
        export_tp4(source, tmp_path / "out", expected_design_sha256=DESIGN_SHA)
    save({**original, "law": "procedural-mcg"})
    with pytest.raises(ValueError, match="contract/design"):
        export_tp4(source, tmp_path / "out", expected_design_sha256=DESIGN_SHA)
    corrupt = json.loads(json.dumps(original))
    corrupt["matrices"][0]["sha256"] = "0" * 64
    save(corrupt)
    with pytest.raises(ValueError, match="SHA-256"):
        export_tp4(source, tmp_path / "out", expected_design_sha256=DESIGN_SHA)
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_resume_rejects_modified_rank_and_changed_source_plan(tmp_path, experts):
    source = write_matrix_fixture(tmp_path / "source", experts)
    target = tmp_path / "ranks"
    result = export_tp4(source, target, expected_design_sha256=DESIGN_SHA)
    path = target / result["entries"][0]["path"]
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="file hash"):
        export_tp4(source, target, expected_design_sha256=DESIGN_SHA, resume=True)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(FileExistsError, match="identical"):
        export_tp4(source, target, expected_design_sha256=DESIGN_SHA, resume=True)
