# Single image for both services (backend API + Streamlit frontend).
# The two compose services run the same image with different commands.
FROM python:3.12-slim

# uv (pinned, official static binary)
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /uvx /bin/

# libgomp1 is needed by onnxruntime (local all-MiniLM-L6-v2 embeddings).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for layer caching (both extras -> one image).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --all-extras --no-dev

# Application code
COPY . .

EXPOSE 8000 8501

# Default command = backend. The frontend service overrides this in compose.
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
