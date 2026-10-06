"""CI step: does the real training code still produce a model that the API can load?

    python -m pipeline.smoke_test --output work/smoke

The unit tests use small synthetic data. This check uses a sample of the REAL
data splits and the real training config, but only 20 trees, so it takes a few
seconds: train -> evaluate -> write a bundle -> load it with FraudScorer (like
the API does) -> score example_request.json. If any of these steps breaks,
fraud-ci is red and the commit is not released to the production jobs.
Nothing is written to MLflow, the database or the models folder.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common import PROJECT_DIR
from scoring import FraudScorer
from train import calibrated_config, evaluate_model, load_config, train_model, use_cpu, write_bundle


def run_smoke_test(output, train_rows=20000, val_rows=10000, trees=20):
    splits = PROJECT_DIR / "data" / "splits"
    train = pd.read_csv(splits / "train.csv", nrows=train_rows, float_precision="round_trip")
    val = pd.read_csv(splits / "val.csv", nrows=val_rows, float_precision="round_trip")

    config = use_cpu(load_config(PROJECT_DIR / "configs" / "training_cpu.json"))
    config["model_params"]["iterations"] = trees
    config["fit_params"] = {"verbose": 0}          # no early stopping, like the retraining
    model = train_model(train, None, config)

    selection = config["threshold_selection"]
    metrics, _, scores = evaluate_model(model, val, config["threshold"], selection, return_predictions=True)
    bundle = Path(output) / "bundle"
    write_bundle(bundle, model, calibrated_config(config, model, metrics["decision_threshold"]), metrics,
                 val, scores, {"model_version": "smoke-test", "run_id": "smoke-test"})

    scorer = FraudScorer(bundle)
    request = json.loads((PROJECT_DIR / "example_request.json").read_text(encoding="utf-8"))
    probabilities, flagged = scorer.score(pd.DataFrame([request["features"]]))
    if not np.all((probabilities >= 0) & (probabilities <= 1)):
        raise RuntimeError(f"The smoke-test model returned {probabilities}, not a probability.")
    # A model that is not better than chance means something in the data or training is broken.
    if metrics["average_precision"] < 2 * float(val["fraud_bool"].mean()):
        raise RuntimeError(f"Average precision {metrics['average_precision']:.3f} is hardly better than chance.")
    return {"average_precision": metrics["average_precision"], "example_probability": float(probabilities[0]),
            "example_flagged": bool(flagged[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "work" / "smoke")
    args = parser.parse_args()
    result = run_smoke_test(args.output)
    print(f"Smoke test passed: validation AP {result['average_precision']:.3f}, "
          f"example probability {result['example_probability']:.4f}")


if __name__ == "__main__":
    main()
