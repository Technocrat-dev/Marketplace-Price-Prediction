# ============================================================
# Marketplace Price Prediction — API Dockerfile
# Multi-stage build for CPU-optimized inference
# ============================================================

# Stage 1: Python dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/

# Create directories for data and outputs
RUN mkdir -p data/raw data/processed outputs/checkpoints

# Copy model checkpoint and preprocessing artifacts
# These are required for inference and must exist locally
COPY outputs/checkpoints/best_model.pt outputs/checkpoints/best_model.pt
COPY outputs/training_results.json outputs/training_results.json
COPY data/processed/metadata.json data/processed/metadata.json
COPY data/processed/name_vocab.json data/processed/name_vocab.json
COPY data/processed/desc_vocab.json data/processed/desc_vocab.json
COPY data/processed/cat_encoder.json data/processed/cat_encoder.json

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the API server
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
