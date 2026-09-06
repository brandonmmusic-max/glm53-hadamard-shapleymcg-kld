import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "scrub_receipts_for_publication", Path(__file__).resolve().parents[1] / "scripts" / "scrub_receipts_for_publication.py")
scrub = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scrub)


def test_known_roots_become_placeholders_most_specific_first(tmp_path: Path):
    campaign = "/media/label/campaign-root"
    text = json.dumps({"command": [f"{scrub.REPO_ROOT}/scripts/x.py", "--design", f"{scrub.REPO_ROOT}/results/d.json"],
                       "sidecars": f"{campaign}/full/sidecars", "scratch": "/home/user/.tool/jobs/ab12cd34/tmp/log.txt",
                       "other": "/home/user/proj/.tool/worktrees/other-branch/file.py", "store": "/media/other/x",
                       "tmp": "/tmp/thing", "plain": "/home/user/models/M"})
    out, counts = scrub.scrub_text(text, {"campaign": campaign})
    data = json.loads(out)
    assert data["command"] == ["<repo>/scripts/x.py", "--design", "<repo>/results/d.json"]
    assert data["sidecars"] == "<campaign>/full/sidecars"
    assert data["scratch"] == "<scratch>/log.txt"
    assert data["other"] == "<worktree:other-branch>/file.py"
    assert data["store"] == "<media>/x" and data["tmp"] == "<tmp>/thing" and data["plain"] == "<home>/models/M"
    # A container-internal home is a real path in runtime manifests, not a personal one.
    container, counts_c = scrub.scrub_text("/root/.cache/huggingface and /rootfs/keep", {})
    assert container == "<container-home>/.cache/huggingface and /rootfs/keep"
    assert counts_c["container_home"] == 1
    assert counts["repo"] == 2 and counts["campaign"] == 1 and counts["scratch"] == 1 and counts["worktree"] == 1


def test_leftover_denied_content_fails_closed(tmp_path: Path):
    word = scrub.DENIED_WORDS[0]
    with pytest.raises(ValueError, match="denied content"):
        scrub.scrub_text(json.dumps({"note": f"driven by {word.title()}"}), {})
    with pytest.raises(ValueError, match="denied content"):
        scrub.scrub_text("job 2a347265", {}, deny=("2a347265",))
    # A denied word that only appears inside a replaced path is fine once the path is gone.
    out, _ = scrub.scrub_text(f"/home/user/.{word}/jobs/abc/tmp/x", {})
    assert out == "<scratch>/x"


def test_cli_writes_copies_and_manifest_or_nothing(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.json").write_text(json.dumps({"path": f"{scrub.REPO_ROOT}/a", "n": 1}))
    (src / "b.txt").write_text("plain text with /media/vol/dir\n")
    out = tmp_path / "out"
    code = scrub.main(["--output-dir", str(out), "--root", "vol=/media/vol", str(src / "a.json") + ":renamed.json", str(src / "b.txt")])
    assert code == 0
    manifest = json.loads((out / scrub.MANIFEST_NAME).read_text())
    assert [f["dest"] for f in manifest["files"]] == ["b.txt", "renamed.json"]
    renamed = json.loads((out / "renamed.json").read_text())
    assert renamed == {"path": "<repo>/a", "n": 1}
    assert (out / "b.txt").read_text() == "plain text with <vol>/dir\n"
    rec = {f["dest"]: f for f in manifest["files"]}
    assert rec["renamed.json"]["source_sha256"] == scrub.sha256_bytes((src / "a.json").read_bytes())
    assert rec["renamed.json"]["scrubbed_sha256"] == scrub.sha256_bytes((out / "renamed.json").read_bytes())
    assert rec["renamed.json"]["source"].startswith("<") and "/home/" not in rec["renamed.json"]["source"]
    # One offending file means no output at all.
    bad = src / "c.json"
    bad.write_text(json.dumps({"note": scrub.DENIED_WORDS[1]}))
    out2 = tmp_path / "out2"
    with pytest.raises(ValueError):
        scrub.main(["--output-dir", str(out2), str(src / "a.json"), str(bad)])
    assert not out2.exists()
