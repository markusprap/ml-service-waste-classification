FROM python:3.11-slim

# Set memory and performance optimizations
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV OMP_NUM_THREADS=2

WORKDIR /app

# Install system dependencies for TensorFlow optimization
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Copy startup script
COPY start.sh .
RUN chmod +x start.sh

# Environment variables for Railway
ENV PORT=8080
ENV FLASK_ENV=production

# Expose port
EXPOSE $PORT

# Use startup script
CMD ["./start.sh"]
