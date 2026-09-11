import numpy as np

from experiments.thickness_variance.ml.common import register, write_json
from experiments.thickness_variance.ml.evaluation import metrics
from experiments.thickness_variance.ml.reporting import report


def test_validation_report_preserves_unverified_execution_limitation(tmp_path):
    model = tmp_path / "models/primary/validation.json"
    validation = tmp_path / "validation_summary.json"
    write_json(model, {
        "scores": [{
            "recipe": "primary", "part": "validation",
            "train_arm": "A", "eval_arm": "A",
            **metrics(np.array([0.15, 0.25]), np.array([0.16, 0.24])),
        }],
        "fits": [{
            "train_arm": "A", "preprocessing_fit_s": 0.01,
            "model_fit_s": 0.02, "selected_features": ["peak_1_time_ns"],
        }],
    })
    limitation = "Historical input-to-output linkage is unverified."
    write_json(validation, {
        "recipes": {"primary": {"artifact": "models/primary/validation.json"}},
        "provenance": {
            "policy": "existing_outputs", "verified_receipts": 0,
            "valid_outputs": 2000, "limitation": limitation,
        },
    })
    register(tmp_path, [model, validation])
    text = report(tmp_path).read_text()
    assert "existing_outputs" in text
    assert "0 of 2000 valid outputs" in text
    assert limitation in text and "Validation only" in text
