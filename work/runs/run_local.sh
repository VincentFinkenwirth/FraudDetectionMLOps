#!/usr/bin/env bash
# One complete simulated year without Docker, in its own folder, so several
# runs (e.g. different audit seeds) can run at the same time.
#
#   bash work/runs/run_local.sh <name> <port> <label mode> [audit seed]
#   bash work/runs/run_local.sh realistic_a 8101 realistic
#   bash work/runs/run_local.sh realistic_b 8102 realistic 1
set -e
cd "$(dirname "$0")/../.."
NAME=$1; PORT=$2; MODE=$3; SEED=$4
RUN=work/runs/$NAME
rm -rf "$RUN"; mkdir -p "$RUN"

set -a; source .env; set +a
export MODEL_DIR=$RUN/models
export DATABASE_URL=sqlite:///$RUN/fraud.db
export MLFLOW_TRACKING_URI=sqlite:///$RUN/mlflow.db
export MLFLOW_ARTIFACT_DIR=$RUN/mlflow_artifacts
export PIPELINE_RUNS_DIR=$RUN/pipeline_runs
export MONITORING_STREAM=$NAME
export MODEL_CHECK_SECONDS=5
export API_URL=http://localhost:$PORT
export MLFLOW_DISABLE_AGENT_HINT=1
export PYTHONWARNINGS=ignore
PY=./.venv-check/Scripts/python.exe

$PY -m training.package_model --models-dir "$MODEL_DIR" > "$RUN/package.log" 2>&1
$PY -m uvicorn service.api:create_app --factory --port "$PORT" > "$RUN/api.log" 2>&1 &
API_PID=$!
for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT/ready" | grep -q ready; then break; fi
    sleep 2
done

SEED_ARG=""
if [ -n "$SEED" ]; then SEED_ARG="--audit-seed $SEED"; fi
$PY -m simulation.replay --pipeline local --label-mode "$MODE" $SEED_ARG --log "$RUN/progress.csv" > "$RUN/replay.log" 2>&1
$PY -m simulation.summarize --output "$RUN/report" > "$RUN/summarize.log" 2>&1
kill $API_PID 2>/dev/null || true
echo "finished $NAME"
