# ==============================================================
# Predictive Maintenance System - Dockerfile
# Multi-purpose image: serves either the Streamlit dashboard or
# the FastAPI REST API, selected via the CMD/entrypoint at runtime.
# ==============================================================
FROM python:3.12-slim

WORKDIR /app

# System dependencies for scientific Python + LightGBM/XGBoost native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-create expected directories (in case models/reports aren't mounted)
RUN mkdir -p models reports data

EXPOSE 8501 8000

# Healthcheck hits the Streamlit healthcheck endpoint by default
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: run the Streamlit dashboard.
# Override at `docker run` time to serve the API instead, e.g.:
#   docker run -p 8000:8000 <image> uvicorn api.main:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
