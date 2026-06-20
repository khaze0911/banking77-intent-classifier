# ---------------------------------------------------------------------------
# Base image: slim Python, no CUDA. Inference runs on CPU.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# ---------------------------------------------------------------------------
# System deps. build-essential only in case a wheel needs to compile.
# Cache cleared in the same layer so it doesn't bloat the image.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Install CPU-only PyTorch FIRST, from the CPU wheel index.
# Critical ordering: if requirements (which list torch) were installed first
# with no index, pip pulls the CUDA build from PyPI (~2.7GB nvidia libs +
# ~700MB triton), baked into a lower layer that a later CPU-torch install
# cannot remove. Installing the CPU wheel up front means CUDA is never fetched.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# ---------------------------------------------------------------------------
# Serving-only dependencies. Training/analysis deps (datasets, evaluate,
# scikit-learn, pandas, matplotlib, seaborn, accelerate, boto3) are excluded;
# they are not needed to serve /predict and only bloat the image.
# ---------------------------------------------------------------------------
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
COPY app/ ./app/

# ---------------------------------------------------------------------------
# Model weights (257MB) baked in so the standalone image runs with a single
# `docker run`. The Kubernetes deployment uses a different, weight-less image
# that downloads weights at startup via an init container.
# ---------------------------------------------------------------------------
COPY models/banking77-distilbert ./models/banking77-distilbert

EXPOSE 8000

# --host 0.0.0.0 so the container is reachable from outside, not just localhost
# inside the container. --workers 1 keeps memory predictable.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]