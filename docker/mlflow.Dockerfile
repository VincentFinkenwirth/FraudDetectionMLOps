# MLflow tracking server with the basic-auth app (login required).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# The [auth] extra installs Flask-WTF, which the basic-auth app needs.
RUN pip install --no-cache-dir "mlflow[auth]>=3,<4"

WORKDIR /mlflow
COPY docker/mlflow_auth.ini /mlflow/auth.ini
RUN useradd --create-home --uid 10002 mlflow && mkdir -p /mlflow/data && chown mlflow /mlflow/data
USER mlflow

EXPOSE 5000
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", "--workers", "2", \
     "--backend-store-uri", "sqlite:////mlflow/data/mlflow.db", \
     "--artifacts-destination", "/mlflow/data/artifacts", \
     "--app-name", "basic-auth"]
