# Initial training and later retraining

The code has two entry points:

- `train.py` trains one fresh CatBoost model with a complete configuration.
- `tune.py` uses Optuna to try different configurations through the same training code.

Both use `prepare_dataset()` from `preprocessing.py`. `prepare_data.py` creates
the CSV splits; it does not do the feature preprocessing.

## Setup

Use Python 3.12 or newer in a working virtual environment, then install
`requirements.txt`. Run the commands below from the project directory.

```powershell
python -m pip install -r requirements.txt
```

Importing the scripts or asking for `--help` does not train anything. The commands
in the next sections **do start training** and are for when training is intended.

## Tune

```powershell
python tune.py --study-name catboost-fpr-v1 --trials 60 --timeout 3600
```

The defaults use the existing 400,000 training rows and 100,000 validation rows.
The test set, simulation pool and reserve are not loaded by either command.
The new default configuration is `configs/training_fpr.json`. Optuna maximises
unweighted validation **recall with FPR strictly below 0.05**. FPR means the
fraction of legitimate applications flagged incorrectly. Recall means the
fraction of fraud caught. This follows the objective in the
[dataset authors' benchmark](https://github.com/feedzai/bank-account-fraud/blob/main/notebooks/empirical_results.ipynb).
It does not maximise precision alone, which could favour very few alerts.

For every fitted model, the code checks all distinct validation score thresholds.
It chooses the highest recall with `FPR < 0.05`; equal-recall thresholds are
resolved in favour of fewer false positives. Tied scores stay together, and
predictions consistently use `score >= threshold`. If no fraud can be caught
under the limit, a finite threshold just above 1 means no alerts. No ROC
interpolation or artificial splitting of tied scores is used.

CatBoost still minimises Logloss and uses unweighted PRAUC for early stopping.
Optuna then compares the resulting fitted models using recall under the FPR
limit. This is a practical GPU-compatible approach, not a guarantee of the best
possible tree count for the FPR objective. Average precision, ROC AUC, precision,
log loss, Brier score and review-budget metrics are still recorded.

`configs/training.json` is unchanged. Passing that older config explicitly keeps
AP as the selection objective and its fixed threshold, but uses the expanded
search space. Use a new study name because the search code has changed.

On GPU, CatBoost rejects an explicit `use_weights` metric option when training
weights are all one. The shared training function removes that option for those
trials. For trials with non-default class weights it keeps `use_weights=False`
so early stopping still uses unweighted PRAUC. The actual model parameters are
saved in the model metadata, and MLflow records `effective_eval_metric`.

PRAUC is calculated on the CPU during GPU training, so CatBoost can print a
CPU-fallback warning. This can slow metric evaluation but does not mean model
training has switched to the CPU.

The expanded search covers:

- Depth: 3 to 8; learning rate: 0.01 to 0.2; L2 regularization: 1 to 60.
- Numeric border count: 128 or 254.
- Class weights: None, SqrtBalanced or Balanced.
- Bayesian sampling with bagging temperature 0 to 3, or Bernoulli sampling with
  subsample 0.66 to 1. Their incompatible settings are never combined.

The first 12 trials of a new study contain four of each class-weight option.
Optuna samples their other parameters, then uses adaptive TPE suggestions. These
12 trials count towards `--trials`; they are not extra. Resuming does not add the
initial queue again. Explicit `class_weights` or `scale_pos_weight` are rejected
by this search because they cannot be combined with automatic class weighting.
They remain available for fixed-configuration training.

The new config permits up to 4,000 trees and waits 150 iterations for PRAUC to
improve. Other settings come from the selected config. Trials run one at a time,
with four CatBoost CPU threads. The timeout stops new trials; an already running
trial is allowed to finish, so total time can exceed it.

Every completed trial has an MLflow run and a local model bundle. The results are
under `work/tuning/catboost-fpr-v1/`, in MLflow experiment `fraud-catboost-fpr`:

- `best_config.json`: the full configuration, selected threshold and threshold
  selection policy to reuse for retraining.
- `best_result.json`: the objective and winning score, AP, precision, recall,
  observed FPR, selected threshold, MLflow run ID and model bundle location.
- `trials.csv`: trial scores and parameters for comparison.
- One folder per MLflow run, containing that trial's model and supporting files.

The winning model is already trained. The script does not train it again after
the search. `best_config.json` includes fixed settings and early stopping, not
only the parameters suggested by Optuna.

In MLflow, compare `validation_recall_at_fpr_limit` (higher is better), then
`validation_precision`, `validation_false_positive_rate` and
`validation_decision_threshold`. Precision and recall now describe the selected
threshold rather than the placeholder 0.5 in the starting config.

Running the command again resumes the named SQLite study and adds more trials.
It refuses to resume if the data, base configuration, training code or recorded
library versions have changed. Use a new `--study-name` in that case. A seeded
sampler makes a fresh search repeatable, but a resumed search is not guaranteed
to propose exactly the same trials as an uninterrupted search.

## Train with fixed settings

For an initial reference model using the supplied defaults:

```powershell
python train.py --run-name catboost-reference
```

For later retraining using the settings selected by Optuna:

```powershell
python train.py --config work/tuning/catboost-fpr-v1/best_config.json --train data/retraining/train.csv --val data/retraining/val.csv --experiment fraud-catboost-fpr --run-name retraining-month-01
```

The retraining paths above are examples for future data; those files do not yet
exist. The caller must select only labels available at that simulated date and
provide a separate later validation set. These commands do not calculate label
availability or create monthly batches. If `source_id` is present, overlapping
or repeated source rows are rejected.

Early stopping can choose a new tree count on each run. Each run saves a fresh
model version. The FPR config also selects a fresh threshold from that run's
validation predictions; it does not blindly reuse the previous model's threshold.
The selected threshold is written to the new bundle's `training_config.json`.
Nothing automatically replaces a deployed model. Validation is
used for early stopping, so final evaluation and deployment decisions need a
separate holdout that has not influenced fitting or model selection.

## Saved model and MLflow

Validation now also produces presentation charts and logs them to the same MLflow
run under `charts/validation`. See [VISUALIZATION.md](VISUALIZATION.md) for the
reusable function, exports from an existing model, and later monthly comparisons.
Pass `--no-charts` to either training command to skip chart generation. Because
Optuna checks training source hashes, use a new study name after these code changes.

Each model bundle contains `model.cbm`, the training configuration, validation
metrics, feature schema, preprocessing source and metadata. The metadata includes
dataset hashes, library versions, the MLflow run ID and the fitted tree count.

The FPR limit is an empirical constraint on the validation set, not a guarantee
for test data or future months. For final evaluation, apply the saved threshold
unchanged to the untouched test set; do not tune it using test labels. During
simulation, monitor the actual false-positive rate once labels arrive.

The older `training.json` uses a fixed threshold of 0.5. The new FPR config
selects its threshold on validation data.
The metrics also include precision and recall when reviewing the top 1%, 2% and
5% of scores. Those ranking metrics resolve ties by input order; they are not
promised review rates for the fixed score threshold.

## Probabilities (class-weight correction)

The selected model uses `SqrtBalanced` class weights, so every fraud row counts
about 10 times during training. The raw `predict_proba()` values were therefore
far too high: on validation the mean score was 8.5 % while the fraud rate is
0.9 %, and the Brier score (0.0273) was worse than always predicting the base
rate (0.0089). The task asks the API for a fraud *probability*, so
`scoring.py` corrects the odds: `p = s / (s + k (1 - s))` with
`k = weight(fraud) / weight(non-fraud)`, which CatBoost stores in the model file.

| Validation | raw score | corrected |
| --- | ---: | ---: |
| Mean score (fraud rate 0.90 %) | 8.49 % | 1.38 % |
| Brier score | 0.0273 | 0.0084 |
| Log loss | 0.111 | 0.038 |
| Threshold | 0.369 | 0.055 |

The correction keeps the order of the scores, so exactly the same applications
are flagged (checked by `package_model.py`) and recall/FPR do not change.
`train.py` now applies it automatically and saves `probability_calibration`
and `threshold_scale: probability` in every bundle. It is a correction for the
class weights, not a fitted calibration, so a change in the fraud rate over
time still shifts the probabilities; monthly retraining updates them.

`package_model.py` turns the tuned bundle (copied to `initial_model/`) into the
deployable version `models/versions/v1` without retraining it.

## CPU training

The tuned configs use `"task_type": "GPU"`. Jenkins and most people who want to
reproduce the project have no GPU, so `train.py --cpu` switches any config to
the CPU, and `configs/training_cpu.json` holds the selected hyperparameters for
the retraining pipeline. The initial model v1 stays the GPU-trained one;
all later versions are trained on the CPU. GPU and CPU CatBoost do not give
bit-identical models, so the two are not directly interchangeable.

## Baseline

```powershell
python baseline.py
```

A logistic regression on the same training split, with the same threshold rule (highest
recall with FPR < 5 % on validation), compared with v1 on validation and test. It needs more
preprocessing than CatBoost (median imputation with a "was missing" indicator, scaling,
one-hot encoding, `class_weight="balanced"`). Results go to `docs/results/baseline.json` and
the MLflow experiment `fraud-baseline`.

| Test set | Logistic regression | CatBoost v1 |
| --- | ---: | ---: |
| Recall at FPR < 5 % | 53.1 % | 57.6 % |
| Average precision | 0.144 | 0.177 |
| ROC AUC | 0.902 | 0.914 |

## Fairness

Evaluation now also reports the false-positive rate for applicants aged 50+
and below 50 (the protected attribute in the BAF benchmark). On validation the
model flags 13.1 % of legitimate applicants aged 50+ but only 3.1 % of younger
ones (ratio 0.23). I did not change the model for this, but the ratio is part of
every evaluation, the promotion report and the weekly monitoring.

For later prediction use `scoring.FraudScorer(bundle_dir).score(data)`. It
applies the matching `prepare_features()`, `predict_proba()` and the
correction. The model logged with the MLflow CatBoost flavour expects prepared
features and returns uncorrected scores.

MLflow defaults to `work/mlflow.db` with artifacts under `work/mlflow_artifacts/`.
To inspect the local runs from the project directory:

```powershell
mlflow ui --backend-store-uri sqlite:///work/mlflow.db
```

The training commands also accept `--tracking-uri` (or `MLFLOW_TRACKING_URI`) and
`--experiment`. With an HTTP tracking server, a new experiment uses the server's
artifact storage. The default setup is intended for local use.

## Checks without training

```powershell
python -m unittest discover -s tests -t . -v
```

The tests make their own small synthetic data (no Base.csv needed) and train
tiny CatBoost models with 30-40 trees. They cover the probability correction,
bundle checks, threshold selection below the FPR limit, drift and fairness
metrics, the REST API (keys, roles, validation, retries, labels, model reload),
the monitoring decisions (incl. the evidence rule for recall), the promotion gate, the
weekly workload threshold, the drift simulation with planned volumes, the baseline, and
one complete retraining run (snapshot to deployment). They do not start a tuning
search. The whole suite takes about 15 seconds.
