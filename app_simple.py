from flask import Flask
from flask_cors import CORS
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
    
    @app.route('/health')
    def health():
        # Simple health check without model loading to avoid startup timeout
        return {
            'status': 'healthy',
            'service': 'ML Classification Service',
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }
    
    # Register API routes with lazy loading
    from src.api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='')
    
    return app

# Create app instance for production (Gunicorn)
app = create_app()

if __name__ == '__main__':
    # For local development only
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
