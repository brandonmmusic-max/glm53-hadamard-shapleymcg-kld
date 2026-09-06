from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/build_p8_mixed_rate_image.py"
MANIFEST_PATH = ROOT / "runtime_patch/p8_mixed_rate_image/image_manifest.json"
DOCKERFILE_PATH = ROOT / "runtime_patch/p8_mixed_rate_image/Dockerfile"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_hashes_every_declared_source() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["schema"] == "glm53.p8-coupled-image-sources.v11"
    for relative, expected in manifest["source_sha256"].items():
        assert _sha(ROOT / relative) == expected


def test_dockerfile_pins_parent_tail_and_actual_import_copies() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    dockerfile = DOCKERFILE_PATH.read_text()
    assert dockerfile.startswith(f"FROM {manifest['parent_image_id']}\n")
    assert manifest["tail_v2"]["sha256"] in dockerfile
    assert "research-only-not-device-qualified" in dockerfile
    assert "decode-m1-prefill-m64-n128-unqualified" in dockerfile
    assert 'org.klc.experiment="glm53-p8-mixed-rate-coupled-k3k4k5-v11"' in dockerfile
    assert "/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels" in dockerfile
    assert "/opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels" in dockerfile
    # Resolve grouped COPY semantics against the exact staged inventory, not
    # string-presence assertions that require one image layer per source file.
    import shlex
    from pathlib import PurePosixPath
    inventory = set(manifest["source_sha256"]) | {
        "runtime_patch/p8_mixed_rate_image/image_manifest.json"
    }
    copied = {}
    for line in dockerfile.splitlines():
        if not line.startswith("COPY "):
            continue
        *sources, destination = shlex.split(line)[1:]
        for source in sources:
            if source.endswith("/"):
                selected = {p: p[len(source):] for p in inventory if p.startswith(source)}
                assert selected
                assert destination.endswith("/")
            else:
                assert source in inventory
                selected = {source: PurePosixPath(source).name}
            for path, suffix in selected.items():
                target = destination + suffix if destination.endswith("/") else destination
                copied.setdefault(path, set()).add(target)
    for source, destinations in manifest["install"].items():
        assert copied[source] == set(destinations)
    for phase in manifest["parent_phase_abi"]["phases"].values():
        assert phase["sha256"] in dockerfile
        for path in phase["paths"]:
            assert path in dockerfile


def test_manifest_pins_repaired_prefill_modules_and_rejects_stale_donors() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    sources = manifest["source_sha256"]
    assert "runtime_patch/p8_coupled_prefill_plan.py" in sources
    assert any(path.endswith("p8_coupled_prefill_fc1.py") for path in sources)
    assert any(path.endswith("p8_coupled_prefill_fc2.py") for path in sources)
    rejected = set(manifest["parent_phase_abi"]["rejected_donor_sha256"])
    assert rejected == {
        "cb72c50dab933ee866103caf7c32f1a6cfb15df991fcdd8b753ac765551946b1",
        "ce31085628a8423468a453275f18c3322f8029d35afccb35f5e7060cbab11a7e",
    }
    assert not rejected.intersection(sources.values())


def test_manifest_encodes_exact_parent_and_candidate_call_abis() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    phases = manifest["parent_phase_abi"]["phases"]
    assert len(phases["phase1"]["call_parameters"]) == 18
    assert phases["phase1"]["call_parameters"][12] == "trellis_rotations"
    assert len(phases["phase2"]["call_parameters"]) == 16
    assert "scale_component" not in phases["phase2"]["call_parameters"]

    candidates = manifest["candidate_launch_abi"]
    assert len(candidates["prefill_fc1"]["call_parameters"]) == 18
    assert candidates["prefill_fc1"]["call_parameters"][12] == "scale_component"
    assert candidates["prefill_fc1"]["constructor_parameters"] == []
    assert candidates["prefill_fc1"]["instance_attributes"]["shared_bytes"] == 65536
    assert len(candidates["prefill_fc2"]["call_parameters"]) == 17
    assert candidates["prefill_fc2"]["call_parameters"][12] == "scale_component"
    assert candidates["prefill_fc2"]["constructor_parameters"] == []
    assert candidates["prefill_fc2"]["instance_attributes"]["shared_bytes"] == 34816


def test_fc1_constructed_resource_contract_proves_disjoint_epilogue() -> None:
    verifier = _load(
        "verify_p8_mixed_rate_image",
        ROOT / "runtime_patch/p8_mixed_rate_image/verify_image.py",
    )
    # The class-level inherited value is 35,328 bytes. The actual no-argument
    # constructor specializes tile_m=64/full_coupled and allocates 65,536.
    instance = SimpleNamespace(
        tile_m=64,
        owned_n=128,
        stage_bytes=17664,
        shared_bytes=65536,
    )
    contract = verifier._fc1_resource_contract(instance)
    assert contract == {
        "pipeline_start": 0,
        "pipeline_end": 35328,
        "gate_fp16_start": 0,
        "gate_fp16_end": 16384,
        "up_fp16_start": 16384,
        "up_fp16_end": 32768,
        "full_output_fp32_start": 32768,
        "full_output_fp32_end": 65536,
        "allocation_end": 65536,
    }
    assert contract["gate_fp16_end"] == contract["up_fp16_start"]
    assert contract["up_fp16_end"] == contract["full_output_fp32_start"]
    assert contract["full_output_fp32_end"] <= contract["allocation_end"]


def test_verifier_checks_phase_hash_abi_owner_and_launch_contract() -> None:
    verifier = (ROOT / "runtime_patch/p8_mixed_rate_image/verify_image.py").read_text()
    for required in (
        'manifest["parent_phase_abi"]',
        'manifest["candidate_launch_abi"]',
        "inspect.signature",
        "inspect.getsource",
        "instance = cls()",
        "_fc1_resource_contract(instance)",
        "stale build/lib donor is present on sys.path",
        "call owner mismatch",
        "constructed attributes mismatch",
        "launch source missing",
    ):
        assert required in verifier


def test_build_is_opt_in_and_uses_offline_immutable_recipe(tmp_path, monkeypatch) -> None:
    builder = _load("build_p8_mixed_rate_image", BUILDER_PATH)
    output = tmp_path / "prepared"
    fake_plan = {
        "status": "prepared",
        "command": ["docker", "build"],
        "execute_requested": False,
    }
    monkeypatch.setattr(builder, "preflight", lambda **_: (output.mkdir(), fake_plan)[1])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("execute must not run without --execute")

    monkeypatch.setattr(builder, "execute", forbidden)
    builder.main(
        [
            "--output",
            str(output),
            "--source-commit",
            "0" * 40,
        ]
    )
    assert json.loads((output / "plan.json").read_text())["status"] == "prepared"

    command = builder.build_command(
        context=tmp_path / "context",
        source_commit="1" * 40,
        source_tree="2" * 40,
        tag=builder.TAG,
    )
    assert command[0:2] == ["docker", "build"]
    assert "--pull=false" in command
    assert command[command.index("--network=none")] == "--network=none"
    assert builder.PARENT in DOCKERFILE_PATH.read_text()
    assert builder.TAG == "klc/glm53-p8-mixed-rate:v11"
    assert builder.INTEGRATION_BASE == "336e082785904e4c0c2bd887fe437ae14adad209"
    assert builder.load_manifest()["runtime_commit"] == builder.INTEGRATION_BASE


def test_manifest_declares_source_tree_import_precedence() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    origins = manifest["required_import_origins"]
    for module, origin in origins.items():
        if module.startswith("b12x."):
            assert origin.startswith("/opt/infernal-invocation/b12x/b12x/")
    assert origins["sitecustomize"] == "/usr/lib/python3.12/sitecustomize.py"
    assert origins["p8_native_kernel"] == "/opt/p8-coupled-runtime/p8_native_kernel.py"
    assert origins["p8_coupled_prefill_plan"] == (
        "/opt/p8-coupled-runtime/p8_coupled_prefill_plan.py"
    )
    assert origins["b12x.moe._shared.kernels.p8_coupled_prefill_fc1"].endswith(
        "/p8_coupled_prefill_fc1.py"
    )
    assert origins["b12x.moe._shared.kernels.p8_coupled_prefill_fc2"].endswith(
        "/p8_coupled_prefill_fc2.py"
    )


def test_mixed_rate_lineage_parameterizes_every_small_m_rate_gate() -> None:
    kernels = ROOT / "runtime_patch/b12x_mixed_rate/b12x/moe/_shared/kernels"
    small_m = (kernels / "p8_small_m.py").read_text()
    assert "P8 small-M FC2 supports K3, K4 or K5 streams" in small_m
    assert "self.b_stage_bytes = 2048 * self.trellis_bits" in small_m
    assert "self.trellis_bits = 4\n" not in small_m
    for name in ("p8_narrow_fc1.py", "p8_h128_fc1.py"):
        text = (kernels / name).read_text()
        assert "self.b_payload_bytes = 1024 * self.trellis_bits" in text
        assert "self.trellis_bits = 4\n" not in text
    dynamic = (kernels / "dynamic.py").read_text()
    assert "and trellis_bits in (3, 4, 5) and trellis_scaled" in dynamic
    assert "trellis_bits == 4 and trellis_scaled" not in dynamic
    assert dynamic.count("trellis_bits=trellis_bits") >= 4
    decoder = (kernels / "w4a8_mcg_decode.py").read_text()
    assert "bits not in (3, 4, 5)" in decoder
    wrapper = (ROOT / "runtime_patch/p8_mixed_rate_image/p8_native_kernel.py").read_text()
    assert 'bits_text not in {"3", "4", "5"}' in wrapper
    assert "K4-only small-M specialization" not in wrapper
    assert "if m != 1 and self.trellis_bits != 4:" in wrapper
    site = (ROOT / "runtime_patch/p8_mixed_rate_image/sitecustomize.py").read_text()
    assert "stream=K{runtime.trellis_bits}" in site and "stream=K4 " not in site
    manifest = json.loads(MANIFEST_PATH.read_text())
    decoder_key = "runtime_patch/b12x_mixed_rate/b12x/moe/_shared/kernels/w4a8_mcg_decode.py"
    assert set(manifest["install"][decoder_key]) == {
        "/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a8_mcg_decode.py",
        "/opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/w4a8_mcg_decode.py",
    }
