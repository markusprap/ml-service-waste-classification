from flask import Flask
from flask_cors import CORS
from src.api.routes import api_bp
from src.config.settings import config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_app():
    app = Flask(__name__)
    CORS(app, origins=config.CORS_ORIGINS)
    
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
    app.register_blueprint(api_bp, url_prefix='')
    
    return app

# Create app instance for production (Gunicorn)
app = create_app()

if __name__ == '__main__':
    # For local development only
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
