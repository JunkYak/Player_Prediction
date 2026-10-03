# ==============================================================================
# NBA Player Performance Prediction System — Production Multi-Stage Dockerfile
# ==============================================================================

# --- Stage 1: Build React Frontend ---
FROM node:20-slim AS frontend-builder
WORKDIR /build

COPY lineup-builder/package*.json ./
RUN npm ci

COPY lineup-builder/ ./
# Build production bundle with relative API path or configurable base URL
ENV VITE_API_BASE_URL=""
RUN npm run build

# --- Stage 2: Production Python Serving Layer & Orchestrator ---
FROM python:3.11-slim AS production

# Prevent Python from writing .pyc files and enable unbuffered streaming logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROPREDICT_DATA_DIR=/app/data \
    PROPREDICT_ENV=production \
    PROPREDICT_STATIC_DIR=/app/static \
    PROPREDICT_SCHEDULE_TIME=05:00

WORKDIR /app

# Install curl for container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application, ML rating engine, scripts, and orchestrators
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY rating_engine/ ./rating_engine/
COPY run_full_pipeline.py .
COPY orchestrator.py .

# Copy compiled frontend assets from Stage 1
COPY --from=frontend-builder /build/dist /app/static

# Create persistent data directory and define volume mount point
RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 8000

# Container liveness health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default entrypoint runs FastAPI serving layer
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
