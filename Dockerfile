# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create necessary directories
RUN mkdir -p /tmp && chmod 777 /tmp

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV PORT=8080
ENV GRADIO_ANALYTICS_ENABLED=False
ENV GRADIO_SHARE=False

# Expose port
EXPOSE 8080

# Add health check endpoint
RUN echo '#!/bin/bash\ncurl -f http://localhost:8080/ || exit 1' > /healthcheck.sh && chmod +x /healthcheck.sh

# Health check with longer timeout for startup
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD /healthcheck.sh

# Run the application with proper signal handling
CMD ["python", "-u", "app.py"]
