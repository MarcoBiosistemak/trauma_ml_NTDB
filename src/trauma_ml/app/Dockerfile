# trauma_ml inference API — Docker image
#
# Build:
#   docker build -f src/trauma_ml/app/Dockerfile -t trauma_ml-api:latest .
#
# Run (mount a model bundle at /models/<id> and point env at it):
#   docker run --rm -p 8000:8000 \
#       -v /path/to/outputs/mortality/models:/models:ro \
#       -e TRAUMA_ML_MODEL_DIR=/models/xgb_0042 \
#       trauma_ml-api:latest
#
# Then test:
#   curl http://localhost:8000/healthz
#   curl http://localhost:8000/metadata | jq
#
FROM python:3.10-slim

# System libs needed by xgboost / lightgbm runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project source (we only need src/ and pyproject.toml at install time)
COPY pyproject.toml /app/
COPY src /app/src

# Install with the [serve] extra — pulls fastapi + uvicorn
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[serve]"

# Round 52: the Doshi FFNN family (model-family 'doshi_ffnn') needs PyTorch at
# INFERENCE time to load/run a saved FFNN artifact.  Install the CPU build for
# serving (much smaller than the CUDA build).  For GPU inference, replace this
# with a CUDA wheel, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Non-root user — recommended for production
RUN useradd --create-home --shell /bin/bash apiuser
USER apiuser

EXPOSE 8000

# Default command — overridable via `docker run ... <new cmd>`
CMD ["uvicorn", "trauma_ml.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
