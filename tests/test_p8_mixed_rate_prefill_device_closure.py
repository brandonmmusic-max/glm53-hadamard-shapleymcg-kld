"""The mixed-rate grouped M64 closure must carry its rate everywhere the K4 one assumed 4.

The M1 closure exercises only the single-token owner, so a K3/K5 layer can pass it while the
grouped prefill owner is wrong. This closure is the gate for batched serving, and every rate
dependent helper it calls has to receive the selected rate rather than a default.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p8_mixed_rate_prefill_device_closure.py"
K4_SCRIPT = ROOT / "scripts/run_p8_coupled_prefill_device_closure.py"


def _module():
    spec = importlib.util.spec_from_file_location("mixed_prefill_closure", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_invocation_is_protocol_only_and_never_touches_cuda() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT)], text=True, capture_output=True, check=True)
    record = json.loads(result.stdout)
    protocol = record["protocol"]
    assert protocol["schema"] == "glm53.p8-mixed-rate-prefill-device-closure.v1"
    assert "K3/K4/K5" in protocol["product"]
    assert protocol["geometry"]["m_cases"] == [2, 64, 65]
    assert protocol["geometry"]["tile_m"] == 64 and protocol["geometry"]["tile_n"] == 128
    assert "--layer" in protocol["geometry"]["layer"]
    # The grouped owner is the point of this closure; the M1 gate is not a substitute.
    assert any("materialized=true M64/N128" in gate for gate in protocol["exact_gates"])


def test_sources_are_the_mixed_rate_kernels_not_the_k4_tree() -> None:
    module = _module()
    for name in ("b12x.moe._shared.kernels.p8_coupled_prefill_fc1",
                 "b12x.moe._shared.kernels.p8_coupled_prefill_fc2",
                 "b12x.moe._shared.kernels.p8_small_m",
                 "b12x.moe._shared.kernels.dynamic"):
        assert "b12x_mixed_rate" in module.MODULE_SOURCES[name], name
        assert "b12x_h16" not in module.MODULE_SOURCES[name], name
    # Every pinned source must exist, or the in-container hash check fails late instead of early.
    for relative in module.MODULE_SOURCES.values():
        assert (ROOT / relative).is_file(), relative


def test_every_rate_dependent_helper_receives_the_selected_rate() -> None:
    text = SCRIPT.read_text()
    assert "M1._decode_expert(sidecar, expert, bits)" in text
    assert "M1._load_sidecar_reference(\n        args.sidecar, args.transform_sha256, args.bits, args.layer\n    )" in text
    assert "_reference_prototypes(args.sidecar, prototypes, scales, args.bits)" in text
    assert "def _reference_prototypes(sidecar: Path, prototypes, scales, bits: int)" in text
    # No bare module-level LAYER may survive; that would pin every run to the pilot layer.
    lines = text.splitlines()
    assert "LAYER = 3" not in lines, "a bare LAYER constant would pin every run to layer 3"
    assert "DEFAULT_LAYER = 3" in lines
    assert not [l for l in lines if "LAYER" in l and "DEFAULT_LAYER" not in l and "--layer" not in l]
    assert "layer=args.layer" in text


def test_probe_invocation_forwards_the_rate_and_layer() -> None:
    module = _module()
    args = module.parse_args([
        "--execute", "--image", "sha256:" + "a" * 64, "--gpu-device", "0",
        "--sidecar", "/tmp/s.safetensors", "--sidecar-sha256", "b" * 64,
        "--design", "/tmp/d.json", "--design-sha256", "c" * 64,
        "--transform", "/tmp/t.json", "--transform-sha256", "d" * 64,
        "--runtime-manifest-sha256", "e" * 64, "--output", "/tmp/out",
        "--bits", "5", "--layer", "29",
    ])
    assert args.bits == 5 and args.layer == 29
    with pytest.raises((SystemExit, ValueError)):
        module.parse_args(["--bits", "6"])


def test_it_is_a_faithful_derivative_of_the_pinned_k4_closure() -> None:
    """Gates and numeric thresholds must not drift from the K4 closure they came from."""
    mixed = _module()
    spec = importlib.util.spec_from_file_location("k4_prefill_closure", K4_SCRIPT)
    k4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k4)
    assert mixed.PROTOCOL["numeric_gates"] == k4.PROTOCOL["numeric_gates"]
    assert mixed.PROTOCOL["exact_gates"] == k4.PROTOCOL["exact_gates"]
    assert mixed.M_CASES == k4.M_CASES
    assert mixed.REPETITIONS == k4.REPETITIONS
    assert mixed.PROTOCOL["failure_policy"] == k4.PROTOCOL["failure_policy"]
    assert mixed.PROTOCOL_SHA256 != k4.PROTOCOL_SHA256, "different sources must hash differently"
