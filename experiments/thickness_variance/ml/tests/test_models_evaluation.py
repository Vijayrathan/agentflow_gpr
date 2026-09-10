import joblib
import numpy as np
import pytest

from experiments.thickness_variance.ml.evaluation import metrics, paired_comparison
from experiments.thickness_variance.ml.models import (
    RF_REFERENCE,
    TorchRegressor,
    fit_pipeline,
    preprocessing,
)


def test_metrics_units_and_delta_sign():
    y = np.array([0.1, 0.2, 0.3])
    score = metrics(y, y + 0.01)
    assert score["rmse_percentage_points"] == pytest.approx(1)
    assert score["bias"] == pytest.approx(0.01)
    result = paired_comparison(
        y, np.array([y, y]), np.array([y + 0.01, y + 0.03]), 1000
    )
    assert result["delta_rmse"] == pytest.approx(0.02)
    assert result["seed_std"] == pytest.approx(np.std([0.01, 0.03], ddof=1))
    assert result["interpretation"] == "A improves"
    same = paired_comparison(y, np.array([y, y]), np.array([y, y]), 1000)
    assert (
        same["ci95"] == [0, 0] and same["interpretation"] == "No detectable difference"
    )


def test_bootstrap_reproducible_and_not_ensemble_metric():
    y = np.array([0.1, 0.2, 0.3])
    a = np.array([y + 0.01, y - 0.01])
    b = np.array([y + 0.02, y - 0.02])
    r = paired_comparison(y, a, b, 1000)
    assert r == paired_comparison(y, a, b, 1000)
    assert r["delta_rmse"] == pytest.approx(0.01)  # ensemble errors would both be zero


def test_nn_exact_architecture_and_reload(tmp_path):
    import torch

    X = np.random.default_rng(4).normal(size=(20, 5))
    y = np.linspace(0.1, 0.3, 20)
    model = TorchRegressor(epochs=2, seed=11).fit(X, y)
    layers = [m for m in model.model_ if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in layers] == [
        (5, 64),
        (64, 64),
        (64, 1),
    ]
    assert len(model.loss_history_) == 2 and np.isfinite(model.predict(X)).all()
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(model.predict(X), joblib.load(path).predict(X))
    repeat = TorchRegressor(epochs=2, seed=11).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), repeat.predict(X))


def test_rf_reference_and_pipeline_reload(tmp_path):
    cfg = {"n_jobs": 1, "device": "cpu", "torch_threads": 1}
    X = np.random.default_rng(4).normal(size=(20, 42))
    y = np.linspace(0.1, 0.3, 20)
    model, _ = fit_pipeline(
        X, y, {"representation": "peaks", "model": "rf"}, RF_REFERENCE, 11, cfg
    )
    assert model.named_steps["model"].n_estimators == 1000
    assert model.named_steps["model"].max_depth == 3
    assert (
        len(model.named_steps["preprocess"].named_steps["select"].selected_names_) == 5
    )
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(model.predict(X), joblib.load(path).predict(X))


@pytest.mark.parametrize(
    "representation,width",
    [("raw", 120), ("pca", 120), ("anova", 120), ("stats", 9), ("peaks", 42)],
)
def test_heldout_transform_never_refits(representation, width):
    X = np.random.default_rng(2).normal(size=(25, width))
    y = np.linspace(0.1, 0.3, 25)
    p = preprocessing(representation, 11, 1).fit(X, y)
    before = p.transform(X).copy()
    p.transform(np.ones((3, width)) * 1e9)
    np.testing.assert_array_equal(p.transform(X), before)
