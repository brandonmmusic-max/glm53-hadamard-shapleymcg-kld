from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p8_mixed_rate_m1_device_closure.py"


def test_carrier_failure_capture_is_exact_bounded_and_create_only(tmp_path):
    import numpy as np
    module = _module()
    actual = torch.tensor([[1, 2, 4]], dtype=torch.uint8)
    expected = torch.tensor([[[1, 3, 4]]], dtype=torch.uint8)
    record = module._save_carrier_failure(tmp_path, {"input": (actual, expected)}, {})
    assert record["gate_changed"] is False
    assert record["carriers"]["input"]["mismatches"] == 1
    assert record["carriers"]["input"]["actual_shape"] == [1, 3]
    assert record["carriers"]["input"]["expected_shape"] == [1, 1, 3]
    assert record["carriers"]["input"]["first_flat_indices"] == [1]
    with np.load(tmp_path / "carrier-failure.npz") as saved:
        assert np.array_equal(saved["input_actual"], actual.numpy())
    with pytest.raises(FileExistsError):
        module._save_carrier_failure(tmp_path, {"input": (actual, expected)}, {})


def _module():
    spec = importlib.util.spec_from_file_location("p8_coupled_m1_closure", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_invocation_is_protocol_only_and_never_touches_cuda() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], text=True, capture_output=True, check=True
    )
    record = json.loads(result.stdout)
    # Protocol revision 2026-09-06: the closure layer became a parameter, because every routed
    # layer has identical geometry and the layer-3 pin was an artifact of the pilot. The prior
    # hash was ae87e8fc90e8113bb9eda84f8a6bac97b9a73e196a894f05fac19cdc807d0781.
    assert record["protocol_sha256"] == (
        "cf2cc3b65dfabf864577f696878ab905ac1aa62b8c14a9137ab93f9131a9043e"
    )
    assert record["protocol"]["schema"] == "glm53.p8-mixed-rate-m1-device-closure.v1"
    assert record["protocol"]["geometry"]["tokens"] == 1
    assert "--layer" in record["protocol"]["geometry"]["layer"]
    assert record["protocol"]["exact_gates"][-1] == (
        "five eager final outputs are bitwise identical"
    )


def test_exact_k32_permutation_is_block_local_and_frozen() -> None:
    module = _module()
    payload = torch.arange(64, dtype=torch.uint8).reshape(1, 64)
    actual = module.permute_k32_payload(payload)
    expected = torch.tensor(
        list(module.K32_PERM) + [32 + value for value in module.K32_PERM],
        dtype=torch.uint8,
    ).reshape(1, 64)
    assert torch.equal(actual, expected)
    assert sorted(actual[0, :32].tolist()) == list(range(32))
    assert sorted(actual[0, 32:].tolist()) == list(range(32, 64))


def test_blocked_quantizer_payload_preserves_exact_wire_bytes_and_rows() -> None:
    module = _module()
    for rows, width in ((1, 4096), (8, 512), (2, 4096)):
        blocked = torch.arange(rows * width).to(torch.uint8).reshape(rows, width // 32, 32)
        actual = module.packed_quantizer_payload(blocked, rows, width)
        expected = module.permute_k32_payload(blocked.reshape(rows, width))
        assert actual.shape == (rows, width)
        assert torch.equal(actual, expected)
        changed = actual.clone()
        changed[0, 0] ^= 1
        assert not torch.equal(changed, expected)
        with pytest.raises(ValueError, match="blocked quantizer"):
            module.packed_quantizer_payload(blocked.reshape(rows, width), rows, width)


def test_materialized_intermediate_reader_uses_slice_major_scale_words() -> None:
    module = _module()
    rows = 32
    carrier = torch.zeros(rows * (128 + 4), dtype=torch.int32)
    byte_view = carrier.view(torch.uint8)
    payload_bytes = rows * 512
    for row in (0, 16):
        byte_view[row * 512 : (row + 1) * 512] = row + 3
        for slice_id in range(4):
            word = rows * 128 + slice_id * rows + row
            byte_view[word * 4 : word * 4 + 4] = torch.tensor(
                [row, slice_id, 127 + slice_id, 255 - row], dtype=torch.uint8
            )
    payload, scales = module.unpack_materialized_intermediate(carrier, [0, 16])
    assert payload.shape == (2, 512)
    assert torch.equal(payload[:, 0], torch.tensor([3, 19], dtype=torch.uint8))
    assert scales.shape == (2, 16)
    assert scales[1].reshape(4, 4)[:, 0].tolist() == [16, 16, 16, 16]
    assert scales[0].reshape(4, 4)[:, 1].tolist() == [0, 1, 2, 3]


def test_signed_nonunit_scale_gate_cannot_pass_identity_or_positive_only() -> None:
    module = _module()
    accepted = module._scale_evidence(
        "test", torch.tensor([-2.0, -0.5, 0.25, 3.0], dtype=torch.float16)
    )
    assert accepted["has_positive"] and accepted["has_negative"]
    with pytest.raises(RuntimeError, match="not signed"):
        module._scale_evidence("identity", torch.ones(16, dtype=torch.float16))
    with pytest.raises(RuntimeError, match="not signed"):
        module._scale_evidence(
            "positive", torch.tensor([0.5, 2.0], dtype=torch.float16)
        )


def test_numerical_gates_are_strict_and_arm_specific() -> None:
    module = _module()
    route = module.PROTOCOL["numeric_gates"]
    assert module._numeric_pass({"cosine": 0.996, "relative_l2": 0.119}, "route")
    assert not module._numeric_pass(
        {
            "cosine": route["route_cosine_strictly_greater_than"],
            "relative_l2": 0.119,
        },
        "route",
    )
    assert not module._numeric_pass(
        {
            "cosine": 0.996,
            "relative_l2": route["final_relative_l2_strictly_less_than"],
        },
        "final",
    )
    with pytest.raises(ValueError, match="unknown numerical arm"):
        module._numeric_pass({"cosine": 1.0, "relative_l2": 0.0}, "aggregate")


def test_actual_wrapper_debug_contract_is_required_not_optional() -> None:
    module = _module()
    wrapper = (ROOT / "runtime_patch/p8_native_kernel.py").read_text()
    assert "self.full_coupled = bool(full_coupled_schema)" in wrapper
    assert "small_m_scheduler=True" not in wrapper  # caller must opt in explicitly.
    assert '"route_output": kernel_output' in wrapper
    assert '"intermediate_u32": intermediate_u32' in wrapper
    assert "P8 scale component requires the M1 N128 owner path" in wrapper
    assert module.REQUIRED_DEBUG == {
        "packed_a", "scale_flat", "intermediate_u32", "route_output",
        "token_map", "row_counts", "expert_tile_base",
    }

def test_every_probe_option_is_forwarded_into_the_container() -> None:
    """A flag added to the parser but not to the container argv silently uses its default.

    That is exactly how the K3 closure ran against layer 3 while being handed a layer-4
    sidecar, so the forwarding is asserted rather than assumed.
    """
    import importlib.util
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location("closure", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.parse_args([
        "--execute", "--image", "sha256:" + "a" * 64, "--gpu-device", "0",
        "--sidecar", "/tmp/s.safetensors", "--sidecar-sha256", "b" * 64,
        "--design", "/tmp/d.json", "--design-sha256", "c" * 64,
        "--transform", "/tmp/t.json", "--transform-sha256", "d" * 64,
        "--runtime-manifest-sha256", "e" * 64, "--output", "/tmp/out",
        "--bits", "3", "--layer", "17",
    ])
    command = module.probe_command(args, _Path("/repo"))
    assert "--probe" in command
    pairs = {command[i]: command[i + 1] for i in range(len(command) - 1) if command[i].startswith("--")}
    assert pairs["--layer"] == "17", "the closure layer must reach the probe"
    assert pairs["--bits"] == "3"
    # Any option the probe itself understands and the outer run knows must be forwarded.
    probe_understands = {"--bits", "--layer", "--image-id", "--sidecar", "--sidecar-sha256",
                         "--design", "--design-sha256", "--transform", "--transform-sha256",
                         "--runtime-manifest", "--runtime-manifest-sha256", "--output"}
    missing = probe_understands - set(pairs)
    assert not missing, f"probe options not forwarded: {sorted(missing)}"
