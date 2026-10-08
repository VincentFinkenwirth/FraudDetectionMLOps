# REST API image: only the code that is needed for scoring, no data and no training code.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
# The lock file pins every package (same versions as the training environment).
COPY requirements-api.lock .
RUN pip install --no-cache-dir -r requirements-api.lock

COPY fraud/__init__.py fraud/preprocessing.py fraud/scoring.py fraud/model_registry.py fraud/
COPY service/ service/

# Do not run as root inside the container.
RUN useradd --create-home --uid 10001 api
USER api

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

# create_app is a factory, so the API keys are read when the server starts.
CMD ["uvicorn", "service.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
