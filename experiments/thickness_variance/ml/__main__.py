"""Run with python -m experiments.thickness_variance.ml COMMAND."""

import argparse
import json
from pathlib import Path

from .common import DEFAULT_CONFIG, load_config, resolve
from .data import audit, snapshot_inputs, split
from .models import RECIPES
from .reporting import report
from .workflow import diagnostics, evaluate, prepare_features, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    commands = parser.add_subparsers(dest="command", required=True)
    s = commands.add_parser(
        "snapshot-inputs",
        help="Freeze configured manifests/decks by arm; does not certify prior execution",
    )
    s.add_argument("--output", required=True, help="New input-hash snapshot path")
    commands.add_parser(
        "audit", help="Read inputs/outputs and emit admission report; exit 2 if blocked"
    )
    commands.add_parser(
        "split",
        help="Lock paired splits after input admission, even if outputs are incomplete",
    )
    f = commands.add_parser(
        "features",
        help="Prepare train/validation features and training-only diagnostic plots",
    )
    f.add_argument(
        "--available-training",
        action="store_true",
        help="Only development plots from available training signals; no trainable cache",
    )
    t = commands.add_parser(
        "train", help="Train recipes, controls and validation predictions; restartable"
    )
    t.add_argument(
        "--recipe",
        action="append",
        choices=list(RECIPES),
        help="Run selected recipe(s); default all",
    )
    e = commands.add_parser(
        "evaluate", help="Freeze fitting, unlock tests, produce paired evaluation"
    )
    e.add_argument(
        "--review-validation",
        action="store_true",
        help="Acknowledge review of validation summary before first test access",
    )
    commands.add_parser("report", help="Build a validation-only or final report")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        if args.command == "snapshot-inputs":
            result = snapshot_inputs(cfg, resolve(args.output))
        elif args.command == "report":
            # Reports summarize bound artifacts without needing to read source datasets.
            result = str(report(Path(cfg["run_dir"])))
        else:
            admission = audit(cfg)
            if args.command == "audit":
                print(
                    json.dumps(
                        {
                            k: admission[k]
                            for k in ("inputs_ready", "outputs_ready", "arms", "provenance")
                        },
                        indent=2,
                    )
                )
                print(
                    f"Issues: {len(admission['issues'])}; details: {cfg['run_dir']}/audit.json"
                )
                for issue in admission["issues"][:10]:
                    print(f"{issue['arm'] or 'A/B'}: {issue['message']}")
                if len(admission["issues"]) > 10:
                    print("Remaining issues are in audit.json")
                for skipped in admission["skipped_checks"]:
                    print(f"Skipped: {skipped}")
                for warning in admission["warnings"]:
                    print(f"Limitation: {warning}")
                return 0 if admission["outputs_ready"] else 2
            splits = split(cfg, admission)
            if args.command == "split":
                result = {k: len(splits[k]) for k in ("train", "validation", "test")}
            elif args.command == "features":
                if not args.available_training:
                    prepare_features(cfg, admission, splits)
                entries = diagnostics(cfg, admission, splits, args.available_training)
                result = {
                    "plotted": sum(e["status"] == "plotted" for e in entries),
                    "available_training_only": args.available_training,
                }
            elif args.command == "train":
                result = train(cfg, admission, splits, args.recipe)
            elif args.command == "evaluate":
                result = evaluate(cfg, admission, splits, args.review_validation)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Blocked: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
