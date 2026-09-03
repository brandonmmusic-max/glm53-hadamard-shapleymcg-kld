import numpy as np

from glm53_nvfp4.paired_role_analysis import bca_mean_interval


def test_bca_mean_interval_contains_observed_for_centered_sample():
    values = np.array([-0.30, -0.20, -0.10, 0.0, 0.10], dtype=np.float64)
    rng = np.random.default_rng(19)
    indices = rng.integers(0, len(values), size=(5000, len(values)))
    interval = bca_mean_interval(values, values[indices].mean(axis=1))
    assert interval[0] < values.mean() < interval[1]
