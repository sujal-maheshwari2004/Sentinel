# =============================================================================
# Sentinel - Multi-stage Dockerfile
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: builder
# Installs dependencies into a venv using uv
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
RUN pip install uv --quiet

# Copy dependency files first so Docker cache is reused when code changes
COPY pyproject.toml uv.lock ./

# Install production dependencies into .venv
RUN uv sync --frozen --no-dev

# -----------------------------------------------------------------------------
# Stage 2: production
# Copies the venv from builder and runs the app
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS production

WORKDIR /app

# Copy the venv from builder
COPY --from=builder /app/.venv .venv

# Copy application code
COPY . .

# Put the venv on PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Ingest port and metrics port
EXPOSE 9000 9001

CMD ["uvicorn", "core.ingestion.server:app", "--host", "0.0.0.0", "--port", "9000"]