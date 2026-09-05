from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/build_p8_coupled_image.py"
MANIFEST_PATH = ROOT / "runtime_patch/p8_coupled_image/image_manifest.json"
DOCKERFILE_PATH = ROOT / "runtime_patch/p8_coupled_image/Dockerfile"


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
    for relative, expected in manifest["source_sha256"].items():
        assert _sha(ROOT / relative) == expected


def test_dockerfile_pins_parent_tail_and_actual_import_copies() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    dockerfile = DOCKERFILE_PATH.read_text()
    assert dockerfile.startswith(f"FROM {manifest['parent_image_id']}\n")
    assert manifest["tail_v2"]["sha256"] in dockerfile
    assert "research-only-not-device-qualified" in dockerfile
    assert "m1-n128-only-unqualified" in dockerfile
    assert "/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels" in dockerfile
    assert "/opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels" in dockerfile
    for destinations in manifest["install"].values():
        for destination in destinations:
            assert destination in dockerfile


def test_build_is_opt_in_and_uses_offline_immutable_recipe(tmp_path, monkeypatch) -> None:
    builder = _load("build_p8_coupled_image", BUILDER_PATH)
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


def test_manifest_declares_source_tree_import_precedence() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    origins = manifest["required_import_origins"]
    for module, origin in origins.items():
        if module.startswith("b12x."):
            assert origin.startswith("/opt/infernal-invocation/b12x/b12x/")
    assert origins["sitecustomize"] == "/usr/lib/python3.12/sitecustomize.py"
    assert origins["p8_native_kernel"] == "/opt/p8-coupled-runtime/p8_native_kernel.py"
