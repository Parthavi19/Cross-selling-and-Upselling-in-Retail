# Multi-stage build for optimized production image
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libfreetype6-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application
COPY app.py .

# Create necessary directories with proper permissions
RUN mkdir -p /tmp/gradio && chmod 777 /tmp/gradio
RUN mkdir -p /app/temp && chmod 755 /app/temp

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV PORT=8080

# Gradio-specific settings to reduce warnings
ENV GRADIO_ANALYTICS_ENABLED=false
ENV GRADIO_SHARE=false
ENV GRADIO_TEMP_DIR=/tmp/gradio
ENV HF_HUB_DISABLE_TELEMETRY=1

# FastAPI settings to reduce deprecation warnings
ENV FASTAPI_ENV=production

# Performance optimizations
ENV OMP_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

# Expose port
EXPOSE 8080

# Health check with extended start period
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Use non-root user for security
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app /tmp/gradio
USER app

# Run the application
CMD ["python", "-u", "app.py"]


