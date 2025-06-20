# Railway Deployment Configuration with optimized settings for ML model
web: gunicorn -w 1 -b 0.0.0.0:$PORT --timeout 300 --max-requests 100 --preload app:app
