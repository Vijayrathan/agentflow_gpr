"""Human-readable validation/test reports with explicitly bounded conclusions."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import read_json, register, verify_artifacts, write_json


def report(run):
    run = Path(run)
    validation = run / "validation_summary.json"
    verify_artifacts(run, [validation])
    summary = read_json(validation)
    scores, fits = [], []
    for name, item in summary["recipes"].items():
        path = run / item["artifact"]
        verify_artifacts(run, [path])
        d = read_json(path)
        scores.extend(d["scores"])
        fits.extend({"recipe": name, **fit} for fit in d["fits"])
    lines = [
        "# Thickness/moisture experiment",
        "",
        "Primary: fixed 1,000-tree, depth-3 RF with five training-selected peak features.",
        "No AGC, label-informed alignment, geometry input, or per-trace normalization.",
        "",
    ]
    provenance = summary.get("provenance", {})
    if provenance:
        lines.extend([
            f"Data admission policy: `{provenance['policy']}`. "
            f"Execution receipts verified for {provenance['verified_receipts']} of "
            f"{provenance['valid_outputs']} valid outputs.",
            "",
        ])
        if provenance.get("limitation"):
            lines.extend([f"**Provenance limitation:** {provenance['limitation']}", ""])
    if (run / "evaluation.json").exists():
        verify_artifacts(run, [run / "evaluation.json"])
        evaluation = read_json(run / "evaluation.json")
        scores.extend(evaluation["scores"])
        main = evaluation["primary_comparisons"]["A"]
        lo, hi = np.asarray(main["ci95"]) * 100
        lines.extend(
            [
                "## Prespecified primary result",
                "",
                (
                    f"{main['interpretation']}: mean ΔRMSE = {main['delta_percentage_points']:.5f} VWC percentage points; "
                    f"paired test-sample 95% interval [{lo:.5f}, {hi:.5f}]. Positive favors A."
                ),
                f"Training-seed standard deviation of ΔRMSE: {main['seed_std'] * 100:.5f} percentage points.",
                "The interval is conditional on fitted runs. Secondary models do not replace a null primary result.",
                "",
            ]
        )
        plots = evaluation_plots(run, evaluation["predictions"])
        lines.extend([f"![{p.stem}]({p.relative_to(run)})" for p in plots])
    else:
        lines.extend(
            [
                "## Validation only",
                "",
                "Test data remain locked. No A/B effectiveness conclusion is available.",
                "",
            ]
        )
        plots = []
    lines.extend(
        [
            "## Scores",
            "",
            "Errors below are VWC percentage points; bias is prediction minus truth.",
            "",
            "| Recipe | Split | Train → evaluate | RMSE mean ± SD | MAE | Bias | R² |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    grouped = defaultdict(list)
    for row in scores:
        grouped[(row["recipe"], row["part"], row["train_arm"], row["eval_arm"])].append(
            row
        )
    for (recipe, part, arm, ev), rows in sorted(grouped.items()):
        values = np.asarray([r["rmse_percentage_points"] for r in rows])
        means = {
            k: float(np.mean([r[k] for r in rows]))
            for k in ("mae_percentage_points", "bias_percentage_points")
        }
        r2 = [r["r2"] for r in rows if r["r2"] is not None]
        r2_text = f"{np.mean(r2):.5f}" if r2 else "undefined"
        lines.append(
            f"| {recipe} | {part} | {arm} → {ev} | {values.mean():.5f} ± {values.std(ddof=1) if len(values) > 1 else 0:.5f} | "
            f"{means['mae_percentage_points']:.5f} | {means['bias_percentage_points']:.5f} | {r2_text} |"
        )
    lines.extend(
        [
            "",
            "## Primary feature-selection stability",
            "",
            "Different selected columns across arms are part of the separately fitted training pipelines.",
            "",
        ]
    )
    stability = {}
    for arm in ("A", "B"):
        count = Counter(
            name
            for fit in fits
            if fit["recipe"] == "primary" and fit["train_arm"] == arm
            for name in fit.get("selected_features", [])
        )
        stability[arm] = dict(sorted(count.items()))
        lines.append(
            f"- {arm}: "
            + "; ".join(f"{k} ({v} fits)" for k, v in sorted(count.items()))
        )
    lines.extend(
        [
            "",
            "## Timing",
            "",
            "| Recipe | Preprocessing fit mean (s) | Model fit mean (s) |",
            "|---|---:|---:|",
        ]
    )
    for recipe in sorted({f["recipe"] for f in fits}):
        rows = [f for f in fits if f["recipe"] == recipe]
        lines.append(
            f"| {recipe} | {np.mean([f['preprocessing_fit_s'] for f in rows]):.4f} | {np.mean([f['model_fit_s'] for f in rows]):.4f} |"
        )
    lines.extend(
        [
            "",
            (
                "Feature extraction times are in features/*.json; inference times are in scores.csv. "
                "Raw/PCA/F-test extraction is identity; PCA/F-test fitting is preprocessing time. "
                "NN artifacts predict on CPU; fitting uses the configured device."
            ),
            "",
            "## Controls and interpretation",
            "",
            (
                "Compare mean_control, dummy_feature_control, and permuted_label_control against primary. "
                "Unexpectedly strong permuted-label performance warrants investigation; it is not automatically proof of leakage."
            ),
            "Training-to-validation gaps are exposed by the split-specific score rows above.",
            "",
            "## Limits",
            "",
            (
                "Only the existing synthetic populations and their shared acquisition are tested. "
                "No new moisture seed, unseen thickness range, lab calibration, learned thickness representation, "
                "convergence, or CPU/GPU equivalence is established. Peak slots describe lobes, not assigned interfaces."
            ),
            "",
        ]
    )
    with (run / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    write_json(run / "selection_stability.json", stability)
    output = run / "REPORT.md"
    output.write_text("\n".join(lines))
    register(
        run, [output, run / "scores.csv", run / "selection_stability.json", *plots]
    )
    return output


def evaluation_plots(run, predictions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in predictions if r["recipe"] == "primary"]
    folder = run / "plots"
    folder.mkdir(exist_ok=True)
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, ev in zip(axes, ("A", "B")):
        for arm, color in (("A", "#237e75"), ("B", "#c46a34")):
            rs = [r for r in rows if r["train_arm"] == arm and r["eval_arm"] == ev]
            ax.scatter(
                [100 * r["theta"] for r in rs],
                [100 * r["prediction"] for r in rs],
                alpha=0.15,
                s=8,
                label=f"Trained on {arm}",
                color=color,
            )
        ax.plot([10, 30], [10, 30], "k--", lw=0.8)
        ax.set(
            title=f"Test geometry {ev}: all fitted seeds",
            xlabel="True VWC (%)",
            ylabel="Predicted VWC (%)",
        )
        ax.legend()
    fig.tight_layout()
    path = folder / "predicted_vs_true.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    for field, label in (
        ("thickness_m", "Effective thickness (m)"),
        ("theta", "True volumetric moisture"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, ev in zip(axes, ("A", "B")):
            for arm, color in (("A", "#237e75"), ("B", "#c46a34")):
                rs = [r for r in rows if r["train_arm"] == arm and r["eval_arm"] == ev]
                ax.scatter(
                    [r[field] for r in rs],
                    [100 * (r["prediction"] - r["theta"]) for r in rs],
                    alpha=0.15,
                    s=8,
                    label=f"Trained on {arm}",
                    color=color,
                )
            ax.axhline(0, color="black", lw=0.8)
            ax.set(
                title=f"Test geometry {ev}",
                xlabel=label,
                ylabel="Error (VWC percentage points)",
            )
            ax.legend()
        fig.tight_layout()
        path = folder / f"error_vs_{field}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
