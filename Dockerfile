# syntax=docker/dockerfile:1.6
# --------------------------------------------------------------------------
# trauma_ml — reproducible container
# Build: docker build -t trauma_ml:0.1 .
# Run  : docker run --rm -it -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" \
#        trauma_ml:0.1 trauma-train --start-id 0 --end-id 0
# --------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Build tools (needed to compile wheels for miceforest, LightGBM, CatBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package metadata first for layer caching
COPY pyproject.toml ./
COPY README.md LICENSE ./
COPY src/ ./src/
COPY config/ ./config/
COPY NTDB_variable_mapping_for_trauma_ML.xlsx ./

# Install with the heavy extras (boosting + automl + imputers + imbalance).
# Adjust to your needs; e.g. "pip install -e .[full]" for everything.
RUN pip install -e ".[boosting,automl,imputers,imbalance,explain]"

# Mount data/ and outputs/ at runtime; keep them out of the image
VOLUME ["/app/data", "/app/outputs"]

CMD ["python", "-c", "import trauma_ml; print(trauma_ml.__version__)"]
