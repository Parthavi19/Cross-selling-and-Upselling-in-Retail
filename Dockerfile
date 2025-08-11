# Use Python 3.11 slim for better memory efficiency
FROM python:3.11-slim

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Optimize Python for production
ENV PYTHONHASHSEED=0
ENV PYTHONOPTIMIZE=2

# Set memory and CPU optimizations
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2
ENV VECLIB_MAXIMUM_THREADS=2
ENV NUMEXPR_NUM_THREADS=2

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy application files
COPY app.py .

# Set proper ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create temp directory for user
RUN mkdir -p /tmp/gradio_tmp && chmod 755 /tmp/gradio_tmp
ENV GRADIO_TEMP_DIR=/tmp/gradio_tmp

# Expose port
EXPOSE 8080

# Health check (removed since Gradio doesn't support /health endpoint natively)
# Instead, rely on Gradio's built-in readiness
# HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
#     CMD curl -f http://localhost:8080/health || exit 1

# Command to run the application
# Removed uvicorn since we're using Gradio directly
CMD ["python", "app.py"]
