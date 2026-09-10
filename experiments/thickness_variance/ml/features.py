"""Deterministic lobe descriptors; no geometry or moisture inputs."""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.utils.validation import check_is_fitted

PEAK_FIELDS = (
    "amplitude",
    "time_ns",
    "prominence",
    "width_ns",
    "area_field_ns",
    "present",
)
PEAK_NAMES = (
    tuple(f"peak_{i}_{field}" for i in range(1, 7) for field in PEAK_FIELDS)
    + tuple(f"gap_{i}_{i + 1}_ns" for i in range(1, 6))
    + ("detected_count",)
)
STAT_NAMES = (
    "minimum",
    "maximum",
    "mean",
    "std",
    "q25",
    "q50",
    "q75",
    "skewness",
    "excess_kurtosis",
)


def extract_peaks(signal, dt_s, f_peak_hz):
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("Expected at least three finite signed field samples")
    if not np.isfinite(dt_s) or not np.isfinite(f_peak_hz) or min(dt_s, f_peak_hz) <= 0:
        raise ValueError("Positive finite timing and source frequency required")
    magnitude = np.abs(x)
    distance = max(1, math.ceil(0.5 / (f_peak_hz * dt_s)))
    minimum = float(0.005 * magnitude.max())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        indices, props = find_peaks(
            magnitude,
            prominence=minimum,
            distance=distance,
            width=(None, None),
            rel_height=0.5,
            wlen=None,
        )
    order = np.lexsort((indices, -props["prominences"]))[:6]
    order = order[np.argsort(indices[order], kind="stable")]
    vector = np.zeros(42, dtype=np.float64)
    ns = dt_s * 1e9
    retained = []
    for slot, pos in enumerate(order):
        idx = int(indices[pos])
        left, right = int(props["left_bases"][pos]), int(props["right_bases"][pos])
        area = float(np.trapezoid(magnitude[left : right + 1], dx=ns))
        vector[slot * 6 : slot * 6 + 6] = [
            x[idx],
            idx * ns,
            props["prominences"][pos],
            props["widths"][pos] * ns,
            area,
            1,
        ]
        retained.append(
            {
                "slot": slot + 1,
                "index": idx,
                "left_base": left,
                "right_base": right,
                "left_ip": float(props["left_ips"][pos]),
                "right_ip": float(props["right_ips"][pos]),
                "width_height": float(props["width_heights"][pos]),
                "prominence": float(props["prominences"][pos]),
                "boundary_base": left == 0 or right == len(x) - 1,
            }
        )
    if len(order) > 1:
        vector[36 : 36 + len(order) - 1] = np.diff(indices[order]) * ns
    vector[-1] = len(indices)
    diagnostics = {
        "candidate_indices": indices.tolist(),
        "retained": retained,
        "no_peaks": len(indices) == 0,
        "zero_signal": bool(not magnitude.any()),
        "flat_adjacent_count": int(np.count_nonzero(np.diff(x) == 0)),
        "minimum_prominence": minimum,
        "minimum_distance_samples": distance,
        "warnings": [str(w.message) for w in caught],
    }
    return vector, diagnostics


def matrix_peaks(X, dt_s, f_peak_hz):
    return np.asarray([extract_peaks(row, dt_s, f_peak_hz)[0] for row in X])


def matrix_stats(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("Expected finite signal matrix")
    std = X.std(axis=1)
    sk, ku = np.zeros(len(X)), np.zeros(len(X))
    nonconstant = std > 0
    sk[nonconstant] = skew(X[nonconstant], axis=1, bias=True)
    ku[nonconstant] = kurtosis(X[nonconstant], axis=1, fisher=True, bias=True)
    return np.column_stack(
        (
            X.min(1),
            X.max(1),
            X.mean(1),
            std,
            np.quantile(X, [0.25, 0.5, 0.75], axis=1).T,
            sk,
            ku,
        )
    )


class FrozenMinMax(TransformerMixin, BaseEstimator):
    """Global or column scaling; training-constant columns always map to zero."""

    def __init__(self, global_scale=False):
        self.global_scale = global_scale

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        axis = None if self.global_scale else 0
        self.minimum_ = X.min(axis=axis)
        self.span_ = X.max(axis=axis) - self.minimum_
        return self

    def transform(self, X):
        check_is_fitted(self, "span_")
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature width changed")
        return np.divide(
            X - self.minimum_, self.span_, out=np.zeros_like(X), where=self.span_ != 0
        )


class RankedSelector(TransformerMixin, BaseEstimator):
    def __init__(self, names=None, k=5, method="rf", seed=11, n_jobs=1):
        self.names, self.k, self.method = names, k, method
        self.seed, self.n_jobs = seed, n_jobs

    def fit(self, X, y):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        self.names_ = (
            list(self.names)
            if self.names is not None
            else [f"t_{i:06d}" for i in range(X.shape[1])]
        )
        if len(self.names_) != X.shape[1] or self.k > X.shape[1]:
            raise ValueError("Selector schema or feature count invalid")
        if self.method == "rf":
            self.estimator_ = RandomForestRegressor(
                n_estimators=1000,
                max_depth=3,
                criterion="squared_error",
                bootstrap=True,
                max_features=1.0,
                min_samples_leaf=1,
                random_state=self.seed,
                n_jobs=self.n_jobs,
            ).fit(X, y)
            self.scores_ = self.estimator_.feature_importances_
        elif self.method == "f":
            self.scores_, self.pvalues_ = f_regression(X, y, force_finite=True)
        else:
            raise ValueError("Unknown ranking method")
        self.indices_ = np.lexsort((np.asarray(self.names_), -self.scores_))[: self.k]
        self.selected_names_ = [self.names_[i] for i in self.indices_]
        return self

    def transform(self, X):
        check_is_fitted(self, "indices_")
        return np.asarray(X)[:, self.indices_]


def peak_plot(signal, dt_s, f_peak_hz, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    _, detail = extract_peaks(signal, dt_s, f_peak_hz)
    t = np.arange(len(signal)) * dt_s * 1e9
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for ax in axes:
        ax.plot(t, signal, lw=0.8, color="#27364b")
        candidates = np.asarray(detail["candidate_indices"], dtype=int)
        ax.scatter(
            t[candidates],
            signal[candidates],
            s=16,
            color="gray",
            label="Detected lobes",
        )
        for p in detail["retained"]:
            i, l, r = p["index"], p["left_base"], p["right_base"]
            sign = 1 if signal[i] >= 0 else -1
            ax.scatter(t[i], signal[i], color="#c75b12", s=24)
            ax.annotate(
                str(p["slot"]),
                (t[i], signal[i]),
                xytext=(3, 4),
                textcoords="offset points",
            )
            ax.vlines(
                t[i], signal[i] - sign * p["prominence"], signal[i], color="#c75b12"
            )
            ax.hlines(
                sign * p["width_height"],
                p["left_ip"] * dt_s * 1e9,
                p["right_ip"] * dt_s * 1e9,
                color="#278275",
            )
            ax.axvspan(t[l], t[r], color="#278275", alpha=0.055)
        ax.set(xlabel="Time (ns)", ylabel="Signed Ez (native field units)")
        ax.grid(alpha=0.15)
    start = min(t[-1] * 0.8, t[int(np.argmax(np.abs(signal)))] + 2 / f_peak_hz * 1e9)
    late = np.asarray(signal)[t >= start]
    limit = max(float(np.max(np.abs(late))), np.finfo(float).tiny) * 1.1
    last_retained = max((t[p["index"]] for p in detail["retained"]), default=start)
    end = min(
        t[-1], max(start + 4 / f_peak_hz * 1e9, last_retained + 2 / f_peak_hz * 1e9)
    )
    axes[1].set(
        xlim=(start, end),
        ylim=(-limit, limit),
        title="Later retained lobes (display zoom only)",
    )
    axes[0].set_title(title + " — lobes, not assigned interfaces")
    axes[1].legend(
        handles=[
            Line2D([], [], color="gray", marker="o", ls="", label="Detected"),
            Line2D([], [], color="#c75b12", marker="o", label="Retained / prominence"),
            Line2D([], [], color="#278275", label="Half-prominence width"),
            Patch(color="#278275", alpha=0.12, label="Area integration bounds"),
        ],
        loc="upper right",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
