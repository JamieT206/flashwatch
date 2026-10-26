"""Leakage tests: features must not change when anything after the as-of cutoff changes."""

import numpy as np
import pandas as pd
import pytest

from flashwatch.features.asof import as_of, history

TIME = pd.date_range("2010-01-01", "2010-12-31", freq="D")


def toy_features(x, issue):
    """Stand-in for features.build: replace with the real builder once it exists."""
    h = history(x, TIME, "era5land", issue, 30)
    return np.stack([h[-1], h[-5:].mean(0), h.mean(0) - h[:10].mean(0)])


def perturb_after(x, cutoff, seed=1):
    y = x.copy()
    after = TIME > cutoff
    y[after] = np.random.default_rng(seed).normal(size=y[after].shape) * 1e3
    return y


def test_cutoff_applies_latency():
    assert as_of("era5land", "2010-06-15") == pd.Timestamp("2010-06-10")
    assert as_of("s2s", "2010-06-15") == pd.Timestamp("2010-06-15")
    with pytest.raises(KeyError):
        as_of("mystery", "2010-06-15")


def test_history_ends_at_cutoff():
    x = np.arange(len(TIME), dtype=float)[:, None]
    h = history(x, TIME, "era5land", "2010-06-15", 10)
    assert len(h) == 10 and TIME[int(h[-1, 0])] == pd.Timestamp("2010-06-10")
    with pytest.raises(ValueError):
        history(x, TIME, "era5land", "2010-01-10", 30)


@pytest.mark.parametrize("issue", ["2010-04-01", "2010-07-15", "2010-10-15"])
def test_features_ignore_the_future(issue):
    x = np.random.default_rng(0).normal(size=(len(TIME), 3, 4))
    f0 = toy_features(x, issue)
    f1 = toy_features(perturb_after(x, as_of("era5land", issue)), issue)
    assert np.array_equal(f0, f1)
