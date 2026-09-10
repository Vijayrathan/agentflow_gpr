import numpy as np
import pytest

from experiments.thickness_variance.ml.features import (
    PEAK_NAMES,
    FrozenMinMax,
    RankedSelector,
    extract_peaks,
    matrix_stats,
)


def extract(x):
    return extract_peaks(np.asarray(x, float), 1e-9, 1e9)


def test_signed_triangle_physical_units_and_missing_slots():
    v, d = extract([0, 1, 2, 1, 0, 0, -2, -4, -2, 0])
    assert len(v) == len(PEAK_NAMES) == 42
    np.testing.assert_allclose(v[:6], [2, 2, 2, 2, 4, 1])
    np.testing.assert_allclose(v[6:12], [-4, 7, 4, 2, 8, 1])
    assert v[36] == 5 and v[-1] == 2
    assert np.count_nonzero(v[12:36]) == 0
    assert np.count_nonzero(v[37:41]) == 0
    assert d["retained"][0]["boundary_base"]


def test_amplitude_scaling_preserves_timing_and_sign():
    x = np.array([0, 1, 2, 1, 0, 0, -3, -6, -3, 0], float)
    a, _ = extract(x)
    b, _ = extract(4 * x)
    np.testing.assert_allclose(b[[0, 2, 4, 6, 8, 10]], 4 * a[[0, 2, 4, 6, 8, 10]])
    np.testing.assert_array_equal(b[[1, 3, 7, 9, 36, 41]], a[[1, 3, 7, 9, 36, 41]])
    c, _ = extract(-x)
    assert c[0] == -a[0] and c[6] == -a[6]


def test_prominence_and_native_distance():
    x = np.zeros(500)
    x[100] = 1000
    x[120] = 100
    x[250] = 4
    x[400] = 6
    v, d = extract_peaks(x, 7.015269213491199e-12, 750e6)
    assert d["minimum_distance_samples"] == 96
    assert d["candidate_indices"] == [100, 400]
    assert v[-1] == 2


def test_six_by_prominence_then_time_and_ties():
    x = np.zeros(40)
    x[np.arange(2, 34, 4)] = [3, 9, 9, 2, 8, 7, 6, 5]
    v, d = extract(x)
    assert v[-1] == 8
    assert [p["index"] for p in d["retained"]] == [6, 10, 18, 22, 26, 30]
    x = np.zeros(40)
    x[np.arange(2, 34, 4)] = 5
    _, d = extract(x)
    assert [p["index"] for p in d["retained"]] == [2, 6, 10, 14, 18, 22]


def test_plateau_midpoint_zero_signal_and_endpoint():
    v, d = extract([0, 1, 3, 3, 1, 0])
    assert d["candidate_indices"] == [2]
    assert d["flat_adjacent_count"] == 1
    v, d = extract(np.zeros(12))
    assert not v.any() and d["no_peaks"] and d["zero_signal"]
    _, d = extract([4, 3, 1, 0])
    assert d["no_peaks"]


def test_overlapping_bases_are_individual_integrals():
    x = np.array([0, 2, 1, 4, 1, 3, 0], float)
    v, d = extract(x)
    intervals = []
    for p in d["retained"]:
        l, r = p["left_base"], p["right_base"]
        intervals.append(set(range(l, r + 1)))
        assert v[(p["slot"] - 1) * 6 + 4] == np.trapezoid(np.abs(x[l : r + 1]), dx=1)
    assert any(a & b for i, a in enumerate(intervals) for b in intervals[i + 1 :])


@pytest.mark.parametrize("x", [[0, np.nan, 1], [0, np.inf, 1], [[1, 2, 3]], [1, 2]])
def test_invalid_signals_rejected(x):
    with pytest.raises(ValueError):
        extract(x)


def test_scaling_is_training_only_and_unclipped():
    tr = np.array([[1.0, 5], [3, 5]])
    scaler = FrozenMinMax().fit(tr)
    np.testing.assert_array_equal(scaler.transform([[5, 99]]), [[2, 0]])
    before = scaler.minimum_.copy()
    scaler.transform([[-200, -500]])
    np.testing.assert_array_equal(scaler.minimum_, before)
    global_scale = FrozenMinMax(global_scale=True).fit(tr)
    np.testing.assert_array_equal(global_scale.transform([[3, 5]]), [[0.5, 1]])


def test_statistics_constants_and_population_std():
    X = np.array([[2, 2, 2, 2], [0, 1, 2, 3]], float)
    stats = matrix_stats(X)
    assert stats.shape == (2, 9)
    assert np.isfinite(stats).all() and stats[0, 7] == stats[0, 8] == 0
    assert stats[1, 3] == np.std(X[1], ddof=0)


def test_selector_train_only_and_tie_order():
    X = np.zeros((20, 6))
    y = np.linspace(0.1, 0.3, 20)
    sel = RankedSelector(names=("z", "e", "d", "c", "b", "a")).fit(X, y)
    assert sel.selected_names_ == ["a", "b", "c", "d", "e"]
    before = sel.scores_.copy()
    assert sel.transform(np.ones((2, 6)) * 1000).shape == (2, 5)
    np.testing.assert_array_equal(before, sel.scores_)
