# Fraud detection — from model to production

IU course DLBDSMTP01, Task 3 (MLOps). A CatBoost REST service prioritizes benefit applications for human review. Predictions are logged, labels arrive four weeks later, and Jenkins monitors drift, adjusts review thresholds, and retrains models.

## 1. Setup

Use **Python 3.12**, **Git**, and **Docker with Compose**. Examples use PowerShell.

```powershell
git clone https://github.com/VincentFinkenwirth/FraudDetectionMLOps.git
cd FraudDetectionMLOps
```

The repository includes the frozen initial model, the prepared data splits, and all 12 months of simulation applications and labels. No Kaggle download, initial model training, or simulation generation is needed for the normal setup. The prepared data occupies approximately **296 MiB** before Git compression. Data sources, transformations, and upstream license notices are documented in [data/README.md](data/README.md).

### Prepare the data and model

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

In `.env`, replace the placeholder API keys, passwords, and `MLFLOW_SECRET_KEY` with random values (API keys need at least 16 characters). Keep the service URLs and repository settings unless your setup differs.

```powershell
python -m training.package_model    # package the supplied model as v1; no training
```

This reads the included validation split, applies the probability correction, and creates `models/versions/v1` and `models/current.json` for the API. Generated deployment bundles remain local and are not tracked in Git.

### Start the Docker stack

```powershell
docker compose up -d --build
```

| Service | URL | Credentials |
| --- | --- | --- |
| API documentation | http://localhost:8000/docs | `X-API-Key` from `.env` |
| Jenkins | http://localhost:8080 | `JENKINS_ADMIN_ID` / `JENKINS_ADMIN_PASSWORD` |
| MLflow | http://localhost:5000 | `admin` / `MLFLOW_ADMIN_PASSWORD` |

Wait for Jenkins's first successful `fraud-ci` build before replaying data. It tests committed code from `GIT_BRANCH` (default: `main`) and releases it to `fraud-monitor` and `fraud-retrain`. Uncommitted changes are not used by those jobs.

To try one prediction, open the API documentation at `http://localhost:8000/docs`, authorize with the predict key from `.env`, and submit the contents of `example_request.json` to `POST /predict`. You do not need to run the full simulated year to try the service.

### Replay the full experiment

```powershell
python -m simulation.replay --pipeline jenkins
docker compose exec -w /project jenkins python -m simulation.summarize --output /reports
```

Reports are written to `work/docker_reports/`. Replay uses all labels by default; add `--label-mode realistic` to label reviewed applications plus a random 5% audit of unreviewed applications. Use `--audit-seed N` to vary the audit sample.

### Optional: run locally without Docker

After preparing the data and model, set these variables to the matching keys in `.env`, then start the API (SQLite and local MLflow are used by default):

```powershell
$env:API_KEY_ADMIN="<admin key>"
$env:API_KEY_PREDICT="<predict key>"
$env:API_KEY_LABELS="<labels key>"
uvicorn service.api:create_app --factory --port 8000
```

In a second terminal with the virtual environment activated:

```powershell
python -m simulation.replay --pipeline local
python -m simulation.summarize
```

Local reports are written to `work/reports/simulation/`.

### Optional: regenerate the prepared data

The included files are sufficient to run the API, Jenkins CI, and the full simulation. To rebuild the simulated months from the included pool, run:

```powershell
python -m simulation.build_months
```

To recreate the original splits, download `Base.csv` from the [Bank Account Fraud Dataset Suite on Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) into `data/raw/` and run:

```powershell
python -m training.get_data         # verify the downloaded source CSV's SHA-256
python -m training.prepare_data
python -m simulation.build_months
```

These commands overwrite the prepared files, so use them only when you intend to regenerate the data. `python -m training.get_data` verifies the source CSV's SHA-256 and creates splits only when `data/splits/split_metadata.json` is absent. The large original CSV stays excluded from Git.

You can also run the unit tests locally:

```powershell
python -m unittest discover -s tests -t .
```

## 2. File structure

| Path | Purpose |
| --- | --- |
| `fraud/` | Shared preprocessing, scoring, drift, fairness, model registry, and lineage code |
| `training/` | Dataset splits, model training, packaging, and evaluation |
| `service/` | FastAPI endpoints, request validation, access control, and database logging |
| `pipeline/` | Monitoring, retraining, promotion, deployment, and smoke tests |
| `simulation/` | Generate and replay the simulated year; summarize results |
| `configs/` | Training, monitoring, retraining, and drift settings |
| `docker/`, `docker-compose.yml` | API, PostgreSQL, MLflow, and Jenkins containers |
| `jenkins/` | CI, monitoring, retraining jobs, and Jenkins configuration |
| `initial_model/` | Frozen initial model included in Git |
| `data/splits/`, `data/simulation/` | Prepared data included in Git for reproduction |
| `data/raw/`, `models/`, `work/` | Local source CSV, generated deployment bundles, and reports; excluded from Git |
| `tests/` | Unit tests |
| `charts/` | Result charts included in Git |

## 3. Findings

The full-label run processed **286,300 applications** across one simulated year, with **52 successful monitoring builds**, **14 trainings**, and **11 model promotions**.

| Approach | Recall | False-positive rate | Sent to review |
| --- | ---: | ---: | ---: |
| Retraining + weekly threshold adjustment | 48.3% | 5.2% | 5.7% |
| Frozen model + weekly threshold adjustment | 46.8% | 5.2% | 5.7% |
| Frozen model + fixed threshold | 53.8% | 7.7% | 8.3% |

- **Threshold adjustment provided most of the benefit:** it restored review workload within one week after the credit-bureau scale change. The fixed-threshold model caught more fraud but required substantially more reviews.
- **Retraining added a modest gain:** +1.5 percentage points of recall at the same review workload with all labels; no gain across three realistic-label runs.
- **Monitoring caught distinct problems:** feature drift exposed the scale change, delayed labels exposed fraudsters' camouflage, and data-quality checks blocked retraining during the mobile-form incident.
- **Key limitations remain:** the 5% audit yielded too few missed-fraud labels to assess recall reliably; legitimate applicants aged 50+ had roughly four times the false-positive rate of younger applicants. BAF is synthetic banking data used as a proxy for benefit applications.

See [charts/](charts/) for visual results. Detailed development notes and result tables in `RESULTS.md` and `docs/` remain local.
