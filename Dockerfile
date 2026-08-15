# ==============================================================================
# Stage 1: Builder Stage (Install PDM & dependencies into virtualenv)
# ==============================================================================
FROM python:3.14-slim AS builder

WORKDIR /app

# Install system build essentials required for LightGBM / C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PDM
RUN pip install --no-cache-dir pdm

# Copy dependency specifications first (for Docker layer caching)
COPY pyproject.toml pdm.lock ./

# Install production dependencies only into .venv
RUN pdm install --prod --no-editable --no-self

# ==============================================================================
# Stage 2: Final Runtime Stage (Minimal footprint)
# ==============================================================================
FROM python:3.14-slim AS runtime

WORKDIR /app

# Install runtime OpenMP shared library (required for LightGBM CPU execution)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Create a non-root user for security best practices
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Copy application source code and models directory
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/

# Expose FastAPI application port
EXPOSE 8000

# Health check using FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Production execution using Gunicorn with Uvicorn worker processes
CMD ["gunicorn", "src.app:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "30", \
     "--access-logfile", "-"]