FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Environment variables
ENV PORT=5000
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Use gunicorn for production with optimized settings
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "120", "--max-requests", "1000", "--max-requests-jitter", "100", "app:app"]
