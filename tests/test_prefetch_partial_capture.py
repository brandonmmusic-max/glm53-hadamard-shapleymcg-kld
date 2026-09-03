import hashlib

import requests

from glm53_nvfp4 import prefetch_partial_capture
from glm53_nvfp4.prefetch_partial_capture import _fetch_range, _reuse_sparse, merge_windows


def test_merge_windows_preserves_only_selected_consecutive_ranges():
    assert merge_windows([7, 3, 4, 4, 11, 12, 13, 20]) == [
        (3, 4),
        (7, 7),
        (11, 13),
        (20, 20),
    ]


def test_fetch_range_retries_hub_rate_limit(monkeypatch):
    limited = requests.Response()
    limited.status_code = 429
    limited.headers["Retry-After"] = "0"
    limited.url = "https://example.test/file"
    success = requests.Response()
    success.status_code = 206
    success._content = b"abcd"
    success.headers["Content-Range"] = "bytes 4-7/16"
    success.url = limited.url

    class Session:
        def __init__(self):
            self.responses = iter((limited, success))

        def get(self, *args, **kwargs):
            return next(self.responses)

    session = Session()
    monkeypatch.setattr(prefetch_partial_capture, "_session", lambda: session)
    monkeypatch.setattr(prefetch_partial_capture.time, "sleep", lambda _: None)

    start, content, digest = _fetch_range(success.url, 4, 7)
    assert start == 4
    assert content == b"abcd"
    assert digest == hashlib.sha256(content).hexdigest()


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
