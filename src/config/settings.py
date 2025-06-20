import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Max file size (16MB)
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    
    # Model settings
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/model-update.h5')
    TARGET_IMAGE_SIZE = int(os.environ.get('TARGET_IMAGE_SIZE', 224))
    
    # Server settings
    HOST = os.environ.get('ML_SERVICE_HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', os.environ.get('ML_SERVICE_PORT', 5000)))  # Railway uses PORT
    DEBUG = os.environ.get('ML_SERVICE_DEBUG', 'False').lower() == 'true'
      # CORS settings - Allow Railway domain and frontend domains
    cors_origins = os.environ.get('CORS_ORIGINS', 'http://localhost:3000,http://localhost:3001,https://backend-waste-classification-production.up.railway.app')
    CORS_ORIGINS = [origin.strip() for origin in cors_origins.split(',')]
    
    # Add Railway domain pattern if in production
    if 'railway.app' in os.environ.get('RAILWAY_PUBLIC_DOMAIN', ''):
        railway_domain = f"https://{os.environ.get('RAILWAY_PUBLIC_DOMAIN')}"
        if railway_domain not in CORS_ORIGINS:
            CORS_ORIGINS.append(railway_domain)
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

config = Config()
