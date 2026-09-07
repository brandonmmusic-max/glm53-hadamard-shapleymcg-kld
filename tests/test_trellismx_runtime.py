import json
from pathlib import Path
import hashlib

from trellismx.cli import main
from trellismx.runtime import runtime_overlay_report


def test_source_checkout_runtime_overlay_hashes_cleanly() -> None:
    report = runtime_overlay_report()
    assert report["status"] == "passed"
    assert report["origin"] == "source-checkout"
    assert report["source_files"] == 15
    assert report["source_mismatches"] == {}
    assert report["executable"] is False
    assert report["dependencies_imported"] is False
    assert report["cuda_used"] is False


def test_runtime_info_cli(tmp_path, capsys) -> None:
    assert main(["runtime-info"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["schema"] == "trellismx.sm120-runtime-report.v1"
    assert value["status"] == "passed"


def test_runtime_overlay_mirror_matches_pinned_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "runtime_overlay/trellismx_runtime_sm120/manifest.json").read_text()
    )
    for relative, expected in manifest["source_sha256"].items():
        source = root / relative
        mirrored = (
            root
            / "runtime_overlay/trellismx_runtime_sm120/overlay"
            / relative
        )
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
        assert hashlib.sha256(mirrored.read_bytes()).hexdigest() == expected
