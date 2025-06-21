from flask import Blueprint, request, jsonify
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
api_bp = Blueprint('api', __name__)

# Global variable untuk lazy loading
_classifier = None

def get_classifier():
    """Lazy load classifier only when needed"""
    global _classifier
    if _classifier is None:
        try:
            from src.models.waste_classifier_robust import WasteClassifier
            _classifier = WasteClassifier()
            logger.info("Robust classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load robust classifier: {e}")
            # Fallback to original classifier
            try:
                from src.models.waste_classifier import WasteClassifier
                _classifier = WasteClassifier()
                logger.info("Fallback classifier loaded successfully")
            except Exception as e2:
                logger.error(f"Failed to load fallback classifier: {e2}")
            raise e
    return _classifier

@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'time': datetime.now().isoformat(),
        'service': 'ML Classification Service',
        'version': '1.0.0'
    })

@api_bp.route('/api/classify', methods=['POST'])
def classify_waste():
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400

        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({
                'success': False,
                'error': 'Empty image file'
            }), 400

        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({
                'success': False,
                'error': 'Empty image content'
            }), 400
        
        # Lazy load classifier here
        try:
            classifier = get_classifier()
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Model loading failed: {str(e)}'
            }), 500
            
        result = classifier.predict(image_bytes)
        
        if not result['success']:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Classification failed')
            }), 500

        return jsonify({
            'success': True,
            'data': {
                'subcategory': result['subcategory'],
                'main_category': result['main_category'],
                'confidence': result['confidence']
            }
        }), 200

    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/api/model-status', methods=['GET'])
def model_status():
    """Check if model is loaded"""
    global _classifier
    try:
        if _classifier is None:
            return jsonify({
                'model_loaded': False,
                'status': 'Model not loaded yet'
            })
        else:
            return jsonify({
                'model_loaded': True,
                'status': 'Model ready for classification'
            })
    except Exception as e:
        return jsonify({
            'model_loaded': False,
            'status': f'Error: {str(e)}'
        }), 500

@api_bp.route('/api/warmup', methods=['GET'])
def warmup():
    """Warmup endpoint to pre-load model"""
    try:
        print("🔥 Warming up model...")
        classifier = get_classifier()
        
        # Test dengan dummy image
        import numpy as np
        from PIL import Image
        import io
        
        # Create small test image
        dummy_img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        dummy_img.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        # Test prediction
        result = classifier.predict(img_bytes)
        
        return jsonify({
            'status': 'Model warmed up successfully',
            'model_loaded': True,
            'test_result': result.get('success', False)
        })
    except Exception as e:
        return jsonify({
            'status': 'Warmup failed',
            'error': str(e),
            'model_loaded': False
        }), 500
