# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /build

# Install compiler dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies into a wheels cache
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download ML models during build stage to prevent startup delays
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; import os; cache_dir = '/build/model_cache'; os.makedirs(cache_dir, exist_ok=True); SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=cache_dir); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', cache_folder=cache_dir); print('Models downloaded successfully!')"

# ==========================================
# Stage 2: Runner
# ==========================================
FROM python:3.10-slim AS runner

WORKDIR /app

# Install runtime dependencies (e.g., curl for Docker health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy user installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /build/model_cache /app/model_cache

# Put installed user packages on path
ENV PATH=/root/.local/bin:$PATH
ENV HF_HOME=/app/model_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache

# Copy application files
COPY app/ /app/app
COPY tests/ /app/tests

# Expose FastAPI port
EXPOSE 8000

# Set environment variables for FastAPI
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
