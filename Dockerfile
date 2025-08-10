FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create temp directory for file uploads
RUN mkdir -p /tmp

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 8080

# Run the Gradio app directly (not with uvicorn)
CMD ["python", "app.py"]
