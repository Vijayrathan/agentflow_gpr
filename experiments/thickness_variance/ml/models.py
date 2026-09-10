"""Fitted transforms and regressors; no metadata is passed to estimators."""

from __future__ import annotations

import os
import time

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from .features import PEAK_NAMES, STAT_NAMES, FrozenMinMax, RankedSelector

RF_REFERENCE = {"n_estimators": 1000, "max_depth": 3}
NN_REFERENCE = {"learning_rate": 0.001, "l2": 0.01}
GRIDS = {
    "rf": [
        {"n_estimators": n, "max_depth": d} for n in (100, 1000) for d in (3, 8, None)
    ],
    "gbr": [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr}
        for n in (100, 300)
        for d in (2, 3)
        for lr in (0.03, 0.1)
    ],
    "svr": [{"C": c, "epsilon": eps} for c in (1, 10, 100) for eps in (0.001, 0.01)],
    "nn": [
        {"learning_rate": lr, "l2": l2}
        for lr in (0.0003, 0.001)
        for l2 in (0, 0.001, 0.01)
    ],
}
RECIPES = {
    "primary": {"representation": "peaks", "model": "rf", "fixed": RF_REFERENCE},
    "nn_reference": {"representation": "raw", "model": "nn", "fixed": NN_REFERENCE},
}
for model in ("rf", "gbr", "svr", "nn"):
    RECIPES[f"raw_{model}"] = {"representation": "raw", "model": model}
for representation in ("stats", "peaks", "anova", "pca"):
    for model in ("rf", "nn"):
        RECIPES[f"{representation}_{model}"] = {
            "representation": representation,
            "model": model,
        }
RECIPES.update(
    {
        "mean_control": {"representation": "dummy", "model": "mean", "fixed": {}},
        "dummy_feature_control": {
            "representation": "dummy",
            "model": "rf",
            "fixed": RF_REFERENCE,
        },
        "permuted_label_control": {
            "representation": "peaks",
            "model": "rf",
            "fixed": RF_REFERENCE,
            "permuted": True,
        },
    }
)


class TorchRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        learning_rate=0.001,
        l2=0.01,
        epochs=100,
        batch_size=16,
        seed=11,
        device="cpu",
        threads=1,
    ):
        self.learning_rate, self.l2 = learning_rate, l2
        self.epochs, self.batch_size, self.seed = epochs, batch_size, seed
        self.device, self.threads = device, threads

    def fit(self, X, y):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        self.n_features_in_ = X.shape[1]
        self.model_ = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=0
        )
        inputs = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        labels = torch.as_tensor(
            np.asarray(y), dtype=torch.float32, device=self.device
        ).reshape(-1, 1)
        generator = torch.Generator().manual_seed(self.seed)
        self.loss_history_ = []
        self.model_.train()
        for _ in range(self.epochs):
            total = 0.0
            order = torch.randperm(len(X), generator=generator)
            for start in range(0, len(X), self.batch_size):
                idx = order[start : start + self.batch_size].to(self.device)
                optimizer.zero_grad()
                mse = torch.nn.functional.mse_loss(
                    self.model_(inputs[idx]), labels[idx]
                )
                penalty = sum(
                    p.square().sum() for p in self.model_.parameters() if p.ndim == 2
                )
                loss = mse + self.l2 * penalty
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite NN loss; no silent fit repair")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(idx)
            self.loss_history_.append(total / len(X))
        self.model_.eval()
        self.model_.cpu()  # portable artifacts and no retained CUDA allocations
        return self

    def predict(self, X):
        import torch

        with torch.no_grad():
            return (
                self.model_(torch.as_tensor(np.asarray(X), dtype=torch.float32))
                .numpy()
                .ravel()
                .astype(float)
            )


def preprocessing(representation, seed, n_jobs):
    steps = []
    if representation in ("peaks", "stats"):
        steps.append(
            (
                "select",
                RankedSelector(
                    PEAK_NAMES if representation == "peaks" else STAT_NAMES,
                    seed=seed,
                    n_jobs=n_jobs,
                ),
            )
        )
    elif representation == "anova":
        steps.append(("select", RankedSelector(k=100, method="f")))
    elif representation == "pca":
        steps.append(("pca", PCA(n_components=2, whiten=False, svd_solver="full")))
    steps.append(("scale", FrozenMinMax(global_scale=representation == "raw")))
    return Pipeline(steps)


def estimator(kind, params, seed, cfg):
    if kind == "rf":
        return RandomForestRegressor(
            **params,
            criterion="squared_error",
            bootstrap=True,
            max_features=1.0,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=cfg["n_jobs"],
        )
    if kind == "gbr":
        return GradientBoostingRegressor(
            **params, loss="squared_error", random_state=seed
        )
    if kind == "svr":
        return SVR(**params, kernel="rbf", gamma="scale")
    if kind == "nn":
        return TorchRegressor(
            **params, seed=seed, device=cfg["device"], threads=cfg["torch_threads"]
        )
    if kind == "mean":
        return DummyRegressor(strategy="mean")
    raise ValueError(f"Unknown estimator {kind}")


def fit_pipeline(X, y, recipe, params, seed, cfg):
    pre = preprocessing(recipe["representation"], seed, cfg["n_jobs"])
    start = time.perf_counter()
    transformed = pre.fit_transform(X, y)
    prep_s = time.perf_counter() - start
    model = estimator(recipe["model"], params, seed, cfg)
    start = time.perf_counter()
    model.fit(transformed, y)
    fit_s = time.perf_counter() - start
    return Pipeline([("preprocess", pre), ("model", model)]), {
        "preprocessing_fit_s": prep_s,
        "model_fit_s": fit_s,
    }


def selection_details(pipeline):
    pre = pipeline.named_steps["preprocess"]
    result = {}
    if "select" in pre.named_steps:
        sel = pre.named_steps["select"]
        result.update(
            selected_features=sel.selected_names_,
            feature_names=sel.names_,
            scores=sel.scores_.tolist(),
        )
    if "pca" in pre.named_steps:
        result["explained_variance_ratio"] = pre.named_steps[
            "pca"
        ].explained_variance_ratio_.tolist()
    if hasattr(pipeline.named_steps["model"], "loss_history_"):
        result["nn_training_loss"] = pipeline.named_steps["model"].loss_history_
    return result
