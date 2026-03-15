# =============================================================================
# Sentinel - Multi-stage Dockerfile
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: builder
# Installs dependencies and generates the protobuf file
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
RUN pip install uv --quiet

# Copy dependency files first so Docker cache is reused when only code changes
COPY pyproject.toml uv.lock ./

# Install all production dependencies including grpcio-tools for proto generation
RUN uv sync --frozen --no-dev
RUN uv add grpcio-tools --no-sync 2>/dev/null || true
RUN .venv/bin/pip install grpcio-tools --quiet

# Copy proto schema and generate sentinel_pb2.py
COPY proto/ ./proto/
RUN .venv/bin/python -m grpc_tools.protoc \
      -I./proto \
      --python_out=. \
      proto/types.proto && \
    mv types_pb2.py sentinel_pb2.py

# -----------------------------------------------------------------------------
# Stage 2: production
# Copies the venv and generated files, runs the app
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS production

WORKDIR /app

# Copy the venv from builder
COPY --from=builder /app/.venv .venv

# Copy generated protobuf file
COPY --from=builder /app/sentinel_pb2.py .

# Copy application code
COPY . .

# Put the venv on PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 9000 9001

CMD ["python", "main.py"]