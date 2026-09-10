"""Stage orchestration with a test lock and restartable per-recipe training."""

from __future__ import annotations

import csv
import itertools
import time
from pathlib import Path

import joblib
import numpy as np

from .common import (
    bind_run,
    read_json,
    register,
    sha256,
    verify_artifacts,
    write_json,
)
from .data import load_signals
from .evaluation import metrics, paired_comparison
from .features import (
    PEAK_NAMES,
    STAT_NAMES,
    extract_peaks,
    matrix_peaks,
    matrix_stats,
    peak_plot,
)
from .models import (
    GRIDS,
    RECIPES,
    estimator,
    fit_pipeline,
    preprocessing,
    selection_details,
)


def ready_run(cfg, audit, splits):
    if not audit["outputs_ready"]:
        raise ValueError(
            "Full workflow blocked: inspect audit.json for missing/invalid outputs or provenance"
        )
    if splits["input_digest"] != audit["input_digest"]:
        raise ValueError("Split refers to different audited inputs")
    run = bind_run(cfg, audit["data_digest"])
    # Splits are regenerated deterministically by the CLI and verified before entry.
    if (run / "artifacts.json").exists() and "splits.json" in read_json(
        run / "artifacts.json"
    ):
        verify_artifacts(run, [run / "splits.json"])
    else:
        register(run, [run / "splits.json"])
    return run


def feature_path(run, arm, split):
    return run / "features" / f"{split}_{arm}.npz"


def prepare_features(cfg, audit, splits, parts=("train", "validation")):
    run = ready_run(cfg, audit, splits)
    created = []
    for part in parts:
        for arm in ("A", "B"):
            path = feature_path(run, arm, part)
            if path.exists():
                verify_artifacts(
                    run, [path, path.with_suffix(".json"), path.with_suffix(".csv")]
                )
                continue
            ids = splits[part]
            x, y, thickness = load_signals(audit, ids, arm)
            start = time.perf_counter()
            peaks = matrix_peaks(
                x, audit["metadata"]["dt_s"], audit["metadata"]["f_peak_hz"]
            )
            peak_s = time.perf_counter() - start
            start = time.perf_counter()
            stats = matrix_stats(x)
            stats_s = time.perf_counter() - start
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                raw=x,
                peaks=peaks,
                stats=stats,
                y=y,
                ids=ids,
                thickness_m=thickness,
            )
            write_json(
                path.with_suffix(".json"),
                {
                    "part": part,
                    "arm": arm,
                    "feature_extraction_s": {
                        "raw": 0,
                        "peaks": peak_s,
                        "stats": stats_s,
                    },
                    "peak_names": PEAK_NAMES,
                    "stat_names": STAT_NAMES,
                    "metadata_columns_excluded_from_X": [
                        "sample_id",
                        "theta",
                        "thickness_m",
                        "arm",
                    ],
                },
            )
            with path.with_suffix(".csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "sample_id",
                        "theta_metadata",
                        "thickness_m_metadata",
                        *PEAK_NAMES,
                        *STAT_NAMES,
                    ]
                )
                for i, sid in enumerate(ids):
                    writer.writerow([sid, y[i], thickness[i], *peaks[i], *stats[i]])
            register(run, [path, path.with_suffix(".json"), path.with_suffix(".csv")])
            created.append(str(path))
    return created


def diagnostic_ids(rows, training_ids):
    eligible = [
        r for r in rows if r["sample_id"] in set(training_ids) and r["arm"] == "A"
    ]
    chosen = set()
    for key in ("theta", "thickness_m"):
        ordered = sorted(eligible, key=lambda r: (r[key], r["sample_id"]))
        if ordered:
            chosen.update(
                ordered[int(i)]["sample_id"]
                for i in np.linspace(0, len(ordered) - 1, 5)
            )
    return sorted(chosen)


def diagnostics(cfg, audit, splits, available_only=False):
    if not available_only and not audit["outputs_ready"]:
        raise ValueError(
            "Use --available-training for explicitly incomplete diagnostic inspection"
        )
    eligible = (
        [r for r in audit["rows"] if r["hdf5_valid"]]
        if available_only
        else audit["rows"]
    )
    ids = diagnostic_ids(eligible, splits["train"])
    folder = Path(cfg["run_dir"]) / (
        "development_diagnostics" if available_only else "diagnostics"
    )
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for arm in ("A", "B"):
        rows = {r["sample_id"]: r for r in audit["rows"] if r["arm"] == arm}
        for sid in ids:
            if sid not in rows or not rows[sid]["hdf5_valid"]:
                entries.append({"arm": arm, "sample_id": sid, "status": "unavailable"})
                continue
            x, _, _ = load_signals(audit, [sid], arm)
            _, detail = extract_peaks(
                x[0], audit["metadata"]["dt_s"], audit["metadata"]["f_peak_hz"]
            )
            path = folder / f"{arm}_{sid}.png"
            peak_plot(
                x[0],
                audit["metadata"]["dt_s"],
                audit["metadata"]["f_peak_hz"],
                f"Training {arm}/{sid}"
                + (" — incomplete development data" if available_only else ""),
                path,
            )
            entries.append(
                {
                    "arm": arm,
                    "sample_id": sid,
                    "status": "plotted",
                    "plot": str(path),
                    **detail,
                }
            )
    write_json(
        folder / "index.json",
        {
            "complete_population": not available_only,
            "no_test_data_used": True,
            "selection": "available training moisture/thickness quantiles"
            if available_only
            else "training moisture/thickness quantiles",
            "entries": entries,
            "input_digest": audit["input_digest"],
            "data_digest": audit["data_digest"],
        },
    )
    return entries


def feature_data(run, arm, part):
    path = feature_path(run, arm, part)
    verify_artifacts(run, [path])
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def X_for(data, representation):
    if representation == "dummy":
        return np.zeros((len(data["y"]), 1))
    return data[representation if representation in ("peaks", "stats") else "raw"]


def tune(cfg, run, name, recipe, data, splits):
    candidates = GRIDS[recipe["model"]]
    log = run / "models" / name / "cv.json"
    if log.exists():
        verify_artifacts(run, [log])
        records = read_json(log)["records"]
    else:
        records = []
    completed = {(r["fold"], r["candidate"]): r for r in records}
    if len(completed) != len(records):
        raise ValueError("Duplicate CV checkpoint")
    for fold, spec in enumerate(splits["folds"]):
        if all((fold, c) in completed for c in range(len(candidates))):
            continue
        transformed, labels = {}, {}
        for arm in ("A", "B"):
            d = data[arm]
            lookup = {int(s): i for i, s in enumerate(d["ids"])}
            tr = [lookup[s] for s in spec["train"]]
            X = X_for(d, recipe["representation"])
            pre = preprocessing(
                recipe["representation"], cfg["split_seed"], cfg["n_jobs"]
            )
            pre.fit(X[tr], d["y"][tr])
            transformed[arm] = {"train": pre.transform(X[tr])}
            for eval_arm in ("A", "B"):
                de = data[eval_arm]
                loc = {int(s): i for i, s in enumerate(de["ids"])}
                ei = [loc[s] for s in spec["heldout"]]
                transformed[arm][eval_arm] = pre.transform(
                    X_for(de, recipe["representation"])[ei]
                )
                labels[eval_arm] = de["y"][ei]
            transformed[arm]["ytrain"] = d["y"][tr]
        for ci, params in enumerate(candidates):
            if (fold, ci) in completed:
                continue
            start = time.perf_counter()
            scores = {}
            for arm in ("A", "B"):
                model = estimator(recipe["model"], params, cfg["split_seed"], cfg)
                model.fit(transformed[arm]["train"], transformed[arm]["ytrain"])
                for ev in ("A", "B"):
                    scores[f"{arm}->{ev}"] = metrics(
                        labels[ev], model.predict(transformed[arm][ev])
                    )["rmse"]
            entry = {
                "fold": fold,
                "candidate": ci,
                "params": params,
                "scores": scores,
                "mean_rmse": float(np.mean(list(scores.values()))),
                "elapsed_s": time.perf_counter() - start,
            }
            records.append(entry)
            write_json(log, {"records": records})
            register(run, [log])
            print(
                f"{name}: fold {fold + 1}/5, candidate {ci + 1}/{len(candidates)}",
                flush=True,
            )
    means = [
        float(np.mean([r["mean_rmse"] for r in records if r["candidate"] == ci]))
        for ci in range(len(candidates))
    ]
    return candidates[int(np.argmin(means))], {"cv_mean_rmse": means}


def prediction_rows(model, X, data, name, train_arm, eval_arm, part, seed):
    start = time.perf_counter()
    predicted = model.predict(X)
    elapsed = time.perf_counter() - start
    score = metrics(data["y"], predicted)
    rows = [
        {
            "recipe": name,
            "train_arm": train_arm,
            "eval_arm": eval_arm,
            "part": part,
            "seed": seed,
            "sample_id": int(s),
            "theta": float(y),
            "prediction": float(p),
            "thickness_m": float(t),
        }
        for s, y, p, t in zip(
            data["ids"], data["y"], predicted, data["thickness_m"], strict=True
        )
    ]
    return rows, {
        "recipe": name,
        "train_arm": train_arm,
        "eval_arm": eval_arm,
        "part": part,
        "seed": seed,
        "inference_s": elapsed,
        **score,
    }


def train(cfg, audit, splits, requested=None):
    run = ready_run(cfg, audit, splits)
    if (run / "test_unlock.json").exists():
        raise ValueError("Tests are unlocked; training is frozen in this run")
    data = {a: feature_data(run, a, "train") for a in ("A", "B")}
    val = {a: feature_data(run, a, "validation") for a in ("A", "B")}
    names = requested or list(RECIPES)
    for name in names:
        recipe = RECIPES[name]
        folder = run / "models" / name
        summary = folder / "validation.json"
        if summary.exists():
            verify_artifacts(run, [summary])
            print(f"{name}: already complete", flush=True)
            continue
        print(f"Training {name}", flush=True)
        params, tuning = (
            (recipe["fixed"], {})
            if "fixed" in recipe
            else tune(cfg, run, name, recipe, data, splits)
        )
        folder.mkdir(parents=True, exist_ok=True)
        seeds = (
            cfg["fit_seeds"]
            if recipe["model"] not in ("svr", "mean")
            else [cfg["fit_seeds"][0]]
        )
        results, predictions, fits = [], [], []
        for seed, arm in itertools.product(seeds, ("A", "B")):
            stem = folder / f"{arm}_{seed}"
            checkpoint = stem.with_suffix(".json")
            if checkpoint.exists():
                verify_artifacts(run, [checkpoint, stem.with_suffix(".joblib")])
                saved = read_json(checkpoint)
            else:
                target = data[arm]["y"].copy()
                if recipe.get("permuted"):
                    target = np.random.default_rng(seed).permutation(target)
                model, timing = fit_pipeline(
                    X_for(data[arm], recipe["representation"]),
                    target,
                    recipe,
                    params,
                    seed,
                    cfg,
                )
                joblib.dump(model, stem.with_suffix(".joblib"))
                restored = joblib.load(stem.with_suffix(".joblib"))
                probe = X_for(data[arm], recipe["representation"])[:3]
                if not np.array_equal(model.predict(probe), restored.predict(probe)):
                    raise ValueError("Model serialization changed predictions")
                saved = {
                    "fit": {
                        "train_arm": arm,
                        "seed": seed,
                        "params": params,
                        **timing,
                        **selection_details(model),
                    },
                    "scores": [],
                    "predictions": [],
                }
                for part, collection in (("train", data), ("validation", val)):
                    for ev in ("A", "B"):
                        rows, score = prediction_rows(
                            model,
                            X_for(collection[ev], recipe["representation"]),
                            collection[ev],
                            name,
                            arm,
                            ev,
                            part,
                            seed,
                        )
                        saved["predictions"].extend(rows)
                        saved["scores"].append(score)
                write_json(checkpoint, saved)
                register(run, [checkpoint, stem.with_suffix(".joblib")])
            predictions.extend(saved["predictions"])
            results.extend(saved["scores"])
            fits.append(saved["fit"])
        write_json(
            summary,
            {
                "recipe": recipe,
                "params": params,
                "tuning": tuning,
                "fits": fits,
                "scores": results,
                "predictions": predictions,
            },
        )
        register(run, [summary])
    summaries = {}
    for name in RECIPES:
        path = run / "models" / name / "validation.json"
        if path.exists():
            verify_artifacts(run, [path])
            scores = [
                r["rmse"]
                for r in read_json(path)["scores"]
                if r["part"] == "validation"
            ]
            summaries[name] = {
                "validation_mean_rmse": float(np.mean(scores)),
                "artifact": str(path.relative_to(run)),
            }
    write_json(
        run / "validation_summary.json",
        {
            "recipes": summaries,
            "complete": set(summaries) == set(RECIPES),
            "primary": "primary",
            "test_data_used": False,
            "training_samples_per_arm": len(splits["train"]),
        },
    )
    register(run, [run / "validation_summary.json"])
    return summaries


def evaluate(cfg, audit, splits, review_validation=False):
    run = ready_run(cfg, audit, splits)
    validation = run / "validation_summary.json"
    verify_artifacts(run, [validation])
    if not read_json(validation)["complete"]:
        raise ValueError(
            "Complete all prescribed recipes/controls before unlocking tests"
        )
    unlock = run / "test_unlock.json"
    evidence = {
        "validation_sha256": sha256(validation),
        "data_digest": audit["data_digest"],
        "primary": "primary",
        "primary_eval_arm": "A",
        "training_frozen": True,
    }
    if not unlock.exists():
        if not review_validation:
            raise ValueError(
                "Review validation_summary.json, then evaluate --review-validation to freeze training and unlock tests"
            )
        write_json(unlock, evidence)
        register(run, [unlock])
    elif read_json(unlock) != evidence:
        raise ValueError("Validation/data changed after test unlock")
    verify_artifacts(run, [unlock])
    prepare_features(cfg, audit, splits, parts=("test",))
    test = {a: feature_data(run, a, "test") for a in ("A", "B")}
    scores, predictions = [], []
    for name, recipe in RECIPES.items():
        folder = run / "models" / name
        verify_artifacts(run, [folder / "validation.json"])
        fitted = read_json(folder / "validation.json")
        for fit in fitted["fits"]:
            arm, seed = fit["train_arm"], fit["seed"]
            path = folder / f"{arm}_{seed}.joblib"
            verify_artifacts(run, [path])
            model = joblib.load(path)  # only locally created, hash-verified artifacts
            for ev in ("A", "B"):
                rows, score = prediction_rows(
                    model,
                    X_for(test[ev], recipe["representation"]),
                    test[ev],
                    name,
                    arm,
                    ev,
                    "test",
                    seed,
                )
                predictions.extend(rows)
                scores.append(score)
    comparisons = {}
    primary_predictions = {
        (r["train_arm"], r["eval_arm"], r["seed"], r["sample_id"]): r["prediction"]
        for r in predictions
        if r["recipe"] == "primary"
    }
    for ev in ("A", "B"):
        matrices = {}
        for arm in ("A", "B"):
            matrices[arm] = [
                [primary_predictions[(arm, ev, seed, sid)] for sid in splits["test"]]
                for seed in cfg["fit_seeds"]
            ]
        comparisons[ev] = paired_comparison(
            test[ev]["y"], matrices["A"], matrices["B"], cfg["bootstrap_samples"]
        )
    result = {
        "scores": scores,
        "primary_comparisons": comparisons,
        "predictions": predictions,
        "primary_endpoint": "RF peaks: delta RMSE on A test geometry",
        "validation_sha256": sha256(validation),
        "data_digest": audit["data_digest"],
    }
    write_json(run / "evaluation.json", result)
    with (run / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    register(run, [run / "evaluation.json", run / "predictions.csv"])
    return comparisons
