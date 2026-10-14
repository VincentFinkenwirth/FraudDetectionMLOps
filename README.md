# Fraud detection — from model to production

IU course DLBDSMTP01, Task 3 

## 1. Setup

Use **Python 3.12**, **Git**, and **Docker**. Start by cloning the repository, then run the remaining commands from its root:

```powershell
git clone https://github.com/VincentFinkenwirth/FraudDetectionMLOps.git
cd FraudDetectionMLOps
```

The repository includes the **frozen initial model**, **prepared data splits**, and **all 12 months of simulation applications and labels**. See [data/README.md](data/README.md) for the data source and attribution.

### Prepare the environment and model

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

This reads the included validation split, applies the probability correction, and creates `models/versions/v1` and `models/current.json` for the API. Generated deployment bundles remain local and are excluded from Git.

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

To try a single prediction, open `http://localhost:8000/docs`, authorize with the predict key from `.env`, and submit the contents of `example_request.json` to `POST /predict`. Running the full simulated year is optional when trying the service.

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

### Optional: regenerate data or run tests

The included files are sufficient for the API, Jenkins CI, and the full experiment. To rebuild the simulated months from the included simulation pool, run:

```powershell
python -m simulation.build_months
```

To recreate the original splits, download `Base.csv` from the [Bank Account Fraud Dataset Suite on Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) into `data/raw/`, then run:

```powershell
python -m training.get_data         # verify the source CSV's SHA-256
python -m training.prepare_data     # recreate the splits
python -m simulation.build_months   # recreate the simulated months
```

Regeneration overwrites the prepared files. `training.get_data` verifies the downloaded CSV but skips split creation when `data/splits/split_metadata.json` already exists; `training.prepare_data` explicitly recreates them. The original CSV remains excluded from Git.

To run the unit tests locally:

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


See [charts/](charts/) for visual results.
