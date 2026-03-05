FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml .

# Install dependencies (no venv, install into system Python)
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy application code
COPY service.py .

EXPOSE 8000

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
