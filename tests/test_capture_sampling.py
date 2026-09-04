import numpy as np

from glm53_nvfp4.capture import LayerCapture, _balanced_route_sample


def test_balanced_route_sample_round_robins_domains() -> None:
    rows = {
        "a": np.asarray([1, 2, 3, 4], dtype=np.int64),
        "b": np.asarray([10, 11, 12, 13], dtype=np.int64),
        "c": np.asarray([20, 21, 22, 23], dtype=np.int64),
    }
    weights = {
        domain: values.astype(np.float32) / 100 for domain, values in rows.items()
    }

    sample = _balanced_route_sample(rows, weights, start=1, count=7)

    assert sample.rows.tolist() == [10, 20, 2, 11, 21, 3, 12]
    assert np.allclose(sample.weights, sample.rows.astype(np.float32) / 100)


def test_balanced_route_sample_fills_after_a_domain_exhausts() -> None:
    rows = {
        "a": np.asarray([1], dtype=np.int64),
        "b": np.asarray([10, 11, 12, 13], dtype=np.int64),
    }
    weights = {domain: np.ones(len(values), dtype=np.float32) for domain, values in rows.items()}

    sample = _balanced_route_sample(rows, weights, start=0, count=5)

    assert sample.rows.tolist() == [1, 10, 11, 12, 13]


def test_capture_window_validation_rejects_sparse_hole() -> None:
    capture = LayerCapture.__new__(LayerCapture)
    capture.rows = 4
    capture.layer = 3
    capture.window_indices = [1]
    capture.hidden_words = np.asarray(
        [[0x3F80, 0], [0, 0], [0, 0], [0, 0]], dtype=np.uint16
    )

    with np.testing.assert_raises_regex(ValueError, "window_index=1"):
        capture._validate_materialized_windows(window_count=2)


def test_capture_window_validation_accepts_nonzero_bf16() -> None:
    capture = LayerCapture.__new__(LayerCapture)
    capture.rows = 4
    capture.layer = 3
    capture.window_indices = [0, 1]
    capture.hidden_words = np.asarray(
        [[0x3F80, 0], [0, 0], [0, 0], [0xBF80, 0]], dtype=np.uint16
    )

    capture._validate_materialized_windows(window_count=2)
