"""Paired metrics and uncertainty in volumetric water content units."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if (
        y.shape != prediction.shape
        or not np.isfinite(prediction).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Invalid predictions or labels")
    error = prediction - y
    rmse = float(np.sqrt(np.mean(error**2)))
    mae, bias = float(np.mean(np.abs(error))), float(np.mean(error))
    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r2": float(r2_score(y, prediction)) if len(y) > 1 and np.var(y) > 0 else None,
        "rmse_percentage_points": 100 * rmse,
        "mae_percentage_points": 100 * mae,
        "bias_percentage_points": 100 * bias,
    }


def paired_comparison(y, pred_a, pred_b, n_bootstrap=10000, seed=2026):
    """Mean of per-fit RMSE differences; never average predictions into an ensemble.

    Both arrays have shape (fit seeds, paired test sample IDs). Bootstrap each
    ID jointly across models/seeds. Training randomness is summarized separately.
    """
    y = np.asarray(y, dtype=float)
    a, b = np.asarray(pred_a, dtype=float), np.asarray(pred_b, dtype=float)
    if (
        a.shape != b.shape
        or a.ndim != 2
        or a.shape[1] != len(y)
        or not np.isfinite([a, b]).all()
    ):
        raise ValueError("Paired prediction matrices differ")
    ea, eb = (a - y) ** 2, (b - y) ** 2
    delta = np.sqrt(eb.mean(1)) - np.sqrt(ea.mean(1))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, 100):
        count = min(100, n_bootstrap - start)
        ix = rng.integers(0, len(y), size=(count, len(y)))
        draws[start : start + count] = (
            np.sqrt(eb[:, ix].mean(2)) - np.sqrt(ea[:, ix].mean(2))
        ).mean(0)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "delta_rmse": float(delta.mean()),
        "delta_percentage_points": float(100 * delta.mean()),
        "per_seed_delta_rmse": delta.tolist(),
        "seed_std": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
        "seed_min": float(delta.min()),
        "seed_max": float(delta.max()),
        "ci95": [float(lo), float(hi)],
        "bootstrap_samples": n_bootstrap,
        "bootstrap_seed": seed,
        "interpretation": "A improves"
        if lo > 0
        else "A degrades"
        if hi < 0
        else "No detectable difference",
        "scope": "Paired test-sample uncertainty conditional on fitted runs; no resampling of training data",
    }
