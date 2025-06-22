#!/bin/bash

# Startup script for Railway deployment
echo "Starting ML Service..."
echo "Port: $PORT"

# Start gunicorn with dynamic port
exec gunicorn \
    --workers=1 \
    --threads=2 \
    --timeout=300 \
    --max-requests=100 \
    --max-requests-jitter=10 \
    --bind=0.0.0.0:${PORT} \
    app:app
