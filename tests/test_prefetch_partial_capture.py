import hashlib

from glm53_nvfp4.prefetch_partial_capture import _reuse_sparse, merge_windows


def test_merge_windows_preserves_only_selected_consecutive_ranges():
    assert merge_windows([7, 3, 4, 4, 11, 12, 13, 20]) == [
        (3, 4),
        (7, 7),
        (11, 13),
        (20, 20),
    ]


def test_reuse_sparse_verifies_materialized_ranges(tmp_path):
    target = tmp_path / "capture.bin"
    target.write_bytes(b"\0" * 64)
    with target.open("r+b") as handle:
        handle.seek(16)
        handle.write(b"fit-range")
    digest = hashlib.sha256(b"fit-range").hexdigest()
    prior = {
        "path": str(target),
        "mode": "sealed-role-sparse-ranges",
        "apparent_bytes": 64,
        "ranges": [{"start": 16, "end": 24, "bytes": 9, "sha256": digest}],
    }
    assert _reuse_sparse(target, 64, prior)["resume_validation"] == (
        "all-materialized-ranges-sha256"
    )
    with target.open("r+b") as handle:
        handle.seek(20)
        handle.write(b"X")
    assert _reuse_sparse(target, 64, prior) is None
