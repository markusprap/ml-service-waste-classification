from flask import Flask
from flask_cors import CORS
from src.api.routes import api_bp
from src.config.settings import config
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_app():
    app = Flask(__name__)
    CORS(app, origins=config.CORS_ORIGINS)
    
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
    app.register_blueprint(api_bp, url_prefix='')
    
    @app.route('/health')
    def health():
        try:
            # Check if model is loaded
            from src.models.waste_classifier import WasteClassifier
            return {
                'status': 'healthy',
                'service': 'ML Classification Service',
                'version': '1.0.0',
                'model_loaded': True,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'service': 'ML Classification Service',
                'version': '1.0.0',
                'model_loaded': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    # For local development only
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
else:
    # For production deployment (Gunicorn will use this)
    app = create_app()
