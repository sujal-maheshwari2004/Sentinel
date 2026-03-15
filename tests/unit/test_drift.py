import numpy as np
import pandas as pd
import pytest


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index between two distributions.
    PSI < 0.1  — no significant change
    PSI 0.1–0.2 — moderate change
    PSI > 0.2  — significant change
    """
    expected_counts, bin_edges = np.histogram(expected, bins=buckets)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Avoid division by zero and log(0)
    expected_pct = np.where(expected_counts == 0, 1e-4, expected_counts / len(expected))
    actual_pct = np.where(actual_counts == 0, 1e-4, actual_counts / len(actual))

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def test_psi_identical_distributions_is_near_zero():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.5, scale=0.1, size=1000)
    psi = _psi(data, data)
    assert psi < 0.01


def test_psi_similar_distributions_is_low():
    rng = np.random.default_rng(42)
    train = rng.normal(loc=0.5, scale=0.1, size=1000)
    live = rng.normal(loc=0.52, scale=0.1, size=1000)   # slight shift
    psi = _psi(train, live)
    assert psi < 0.1


def test_psi_different_distributions_is_high():
    rng = np.random.default_rng(42)
    train = rng.normal(loc=0.2, scale=0.05, size=1000)
    live = rng.normal(loc=0.8, scale=0.05, size=1000)   # large shift
    psi = _psi(train, live)
    assert psi > 0.2


def test_psi_is_nonnegative():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, 500)
    b = rng.uniform(0, 1, 500)
    assert _psi(a, b) >= 0


def test_psi_warning_threshold():
    """Drift above 0.2 should trigger a warning in real usage."""
    rng = np.random.default_rng(99)
    train = rng.normal(loc=0.3, scale=0.05, size=2000)
    live = rng.normal(loc=0.7, scale=0.05, size=2000)
    psi = _psi(train, live)

    warning_threshold = 0.2
    retrain_threshold = 0.4

    assert psi > warning_threshold
    assert psi > retrain_threshold