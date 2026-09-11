import numpy as np
import pytest

from experiments.thickness_variance.ml import workflow
from experiments.thickness_variance.ml.common import read_json, sha256, write_json
from experiments.thickness_variance.ml.data import make_splits
from experiments.thickness_variance.ml.reporting import report


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    # Explicitly synthetic unit-test fixture. No production output is trained here.
    ids = np.arange(1, 101)
    y = np.linspace(0.101, 0.299, 100)
    splits = make_splits(ids, y)
    splits["input_digest"] = "unit-test-inputs"
    cfg = {
        "run_dir": str(tmp_path),
        "n_jobs": 1,
        "device": "cpu",
        "torch_threads": 1,
        "fit_seeds": [11, 22],
        "split_seed": 2026,
        "bootstrap_samples": 200,
    }
    admission = {
        "outputs_ready": True,
        "input_digest": "unit-test-inputs",
        "data_digest": "unit-test-data",
        "metadata": {"dt_s": 1e-10, "f_peak_hz": 750e6},
        "rows": [],
        "provenance": {
            "policy": "existing_outputs", "valid_outputs": 200,
            "verified_receipts": 0, "unverified_outputs": 200,
            "historical_execution_verified": False,
            "limitation": "Synthetic fixture: execution provenance is unverified.",
        },
    }
    t = np.arange(128)

    def load(_audit, selected, arm):
        theta = y[np.asarray(selected) - 1]
        delay = 30 + (
            np.asarray(selected) % 8 if arm == "A" else np.zeros(len(selected))
        )
        x = np.stack(
            [
                (1 + v) * np.exp(-(((t - d) / 4) ** 2))
                - v * np.exp(-(((t - d - 20) / 5) ** 2))
                for v, d in zip(theta, delay, strict=True)
            ]
        )
        thickness = 0.21 + (
            0.005 * (np.asarray(selected) % 8)
            if arm == "A"
            else np.zeros(len(selected))
        )
        return x, theta, thickness

    monkeypatch.setattr(workflow, "load_signals", load)
    chosen = {k: workflow.RECIPES[k] for k in ("primary", "mean_control", "raw_svr")}
    monkeypatch.setattr(workflow, "RECIPES", chosen)
    monkeypatch.setitem(
        workflow.GRIDS, "svr", [{"C": 1, "epsilon": 0.001}, {"C": 10, "epsilon": 0.001}]
    )
    write_json(tmp_path / "splits.json", splits)
    return cfg, admission, splits


def test_full_workflow_test_lock_resume_and_reporting(experiment):
    cfg, admission, splits = experiment
    run = workflow.ready_run(cfg, admission, splits)
    workflow.prepare_features(cfg, admission, splits)
    assert not workflow.feature_path(run, "A", "test").exists()
    summaries = workflow.train(cfg, admission, splits)
    assert set(summaries) == {"primary", "mean_control", "raw_svr"}
    assert read_json(run / "validation_summary.json")["complete"]
    assert read_json(run / "validation_summary.json")["provenance"] == admission["provenance"]
    assert "Validation only" in report(run).read_text()
    checkpoint = run / "models/primary/A_11.joblib"
    saved_hash = sha256(checkpoint)
    workflow.train(cfg, admission, splits)
    assert sha256(checkpoint) == saved_hash
    with pytest.raises(ValueError, match="Review validation"):
        workflow.evaluate(cfg, admission, splits)
    assert not workflow.feature_path(run, "A", "test").exists()
    comparisons = workflow.evaluate(cfg, admission, splits, review_validation=True)
    assert read_json(run / "evaluation.json")["provenance"] == admission["provenance"]
    assert set(comparisons) == {"A", "B"}
    assert comparisons == workflow.evaluate(cfg, admission, splits)
    text = report(run).read_text()
    assert admission["provenance"]["limitation"] in text
    assert (
        "Prespecified primary result" in text
        and "Secondary models do not replace a null" in text
    )
    assert (run / "predictions.csv").exists() and (
        run / "plots/error_vs_thickness_m.png"
    ).exists()
    with pytest.raises(ValueError, match="training is frozen"):
        workflow.train(cfg, admission, splits)


def test_corrupt_cache_and_changed_data_fail_closed(experiment):
    cfg, admission, splits = experiment
    workflow.prepare_features(cfg, admission, splits)
    run = workflow.ready_run(cfg, admission, splits)
    path = workflow.feature_path(run, "A", "train")
    with path.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed registered artifact"):
        workflow.feature_data(run, "A", "train")
    with pytest.raises(ValueError, match="Run inputs"):
        workflow.ready_run(cfg, {**admission, "data_digest": "changed"}, splits)


def test_incomplete_data_cannot_prepare_or_train(experiment):
    cfg, admission, splits = experiment
    admission["outputs_ready"] = False
    with pytest.raises(ValueError, match="Full workflow blocked"):
        workflow.prepare_features(cfg, admission, splits)
    with pytest.raises(ValueError, match="Full workflow blocked"):
        workflow.train(cfg, admission, splits)


def test_only_training_ids_chosen_for_peak_diagnostics():
    rows = [
        {
            "arm": "A",
            "sample_id": i,
            "theta": 0.1 + i / 1000,
            "thickness_m": 0.3 - i / 1000,
        }
        for i in range(1, 101)
    ]
    chosen = workflow.diagnostic_ids(rows, [1, 2, 3, 4, 5, 6, 7])
    assert set(chosen) <= set(range(1, 8))


def test_cv_preprocessing_fits_only_fold_training_rows(experiment, monkeypatch):
    cfg, admission, splits = experiment
    run = workflow.ready_run(cfg, admission, splits)
    workflow.prepare_features(cfg, admission, splits)
    data = {arm: workflow.feature_data(run, arm, "train") for arm in ("A", "B")}
    original = workflow.preprocessing
    fitted_labels = []

    class ObservedTransform:
        def __init__(self, *args):
            self.transformer = original(*args)

        def fit(self, X, y):
            fitted_labels.append(tuple(y))
            self.transformer.fit(X, y)
            return self

        def transform(self, X):
            return self.transformer.transform(X)

    monkeypatch.setattr(workflow, "preprocessing", ObservedTransform)
    recipe = {"representation": "raw", "model": "svr"}
    workflow.tune(cfg, run, "cv_fit_scope", recipe, data, splits)
    lookup = {int(i): y for i, y in zip(data["A"]["ids"], data["A"]["y"], strict=True)}
    expected = [
        tuple(lookup[i] for i in fold["train"])
        for fold in splits["folds"]
        for _ in ("A", "B")
    ]
    assert fitted_labels == expected
