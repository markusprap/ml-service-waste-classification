#!/usr/bin/env python3
"""
Improved waste_classifier with better Railway compatibility
"""

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import io

class WasteClassifier:
    def __init__(self):
        # Model path priority
        self.model_candidates = [
            "model-trained.h5",      # Our trained model (highest priority)
            "model-simple.h5",       # Simple compatible model
            "model-update.h5"        # Original fallback
        ]
        
        self.model = None
        self.target_size = (224, 224)
        self.classes = None
        self.kategori_mapping = {
            "Sisa_Buah_dan_Sayur": "Organik",
            "Sisa_Makanan": "Organik",
            "Alumunium": "Anorganik",
            "Kaca": "Anorganik",
            "Kardus": "Anorganik",
            "Karet": "Anorganik",
            "Kertas": "Anorganik",
            "Plastik": "Anorganik",
            "Styrofoam": "Anorganik",
            "Tekstil": "Anorganik",
            "Alat_Pembersih_Kimia": "B3",
            "Baterai": "B3",
            "Lampu_dan_Elektronik": "B3",
            "Minyak_dan_Oli_Bekas": "B3",
            "Obat_dan_Medis": "B3"
        }

        # Set environment for CPU only
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def _load_model(self):
        """Load the ML model with robust error handling"""
        if self.model is not None:
            return

        print("🔍 Loading model with improved compatibility...")
        
        for model_name in self.model_candidates:
            model_path = os.path.join(os.path.dirname(__file__), model_name)
            
            if not os.path.exists(model_path):
                print(f"⏭️ {model_name} not found, trying next...")
                continue
                
            print(f"📂 Trying to load {model_name}...")
            
            # Try multiple loading methods
            loading_methods = [
                ("Standard loading", self._load_standard),
                ("Without compilation", self._load_no_compile),
                ("With custom objects", self._load_custom_objects),
                ("Compatibility mode", self._load_compatibility_mode)
            ]
            
            for method_name, method_func in loading_methods:
                try:
                    print(f"  🔧 {method_name}...")
                    self.model = method_func(model_path)
                    
                    if self.model is not None:
                        print(f"  ✅ Success with {method_name}!")
                        
                        # Validate model
                        if self._validate_model():
                            print(f"✅ Model {model_name} loaded and validated successfully!")
                            return
                        else:
                            print(f"  ❌ Model validation failed")
                            self.model = None
                            
                except Exception as e:
                    print(f"  ❌ {method_name} failed: {str(e)[:100]}...")
                    continue
            
            print(f"❌ All methods failed for {model_name}")
        
        raise Exception("❌ Could not load any model. Please check model files.")

    def _load_standard(self, model_path):
        """Standard TensorFlow model loading"""
        return tf.keras.models.load_model(model_path)

    def _load_no_compile(self, model_path):
        """Load without compilation"""
        return tf.keras.models.load_model(model_path, compile=False)

    def _load_custom_objects(self, model_path):
        """Load with custom objects for compatibility"""
        custom_objects = {
            'InputLayer': tf.keras.layers.InputLayer,
            'Dense': tf.keras.layers.Dense,
            'Conv2D': tf.keras.layers.Conv2D,
            'MaxPooling2D': tf.keras.layers.MaxPooling2D,
            'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
            'Flatten': tf.keras.layers.Flatten,
            'Dropout': tf.keras.layers.Dropout,
            'BatchNormalization': tf.keras.layers.BatchNormalization,
            'ReLU': tf.keras.layers.ReLU,
            'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D
        }
        return tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)

    def _load_compatibility_mode(self, model_path):
        """Load with maximum compatibility settings"""
        # Set TF to be more permissive
        tf.compat.v1.disable_eager_execution()
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            tf.compat.v1.enable_eager_execution()
            return model
        except:
            tf.compat.v1.enable_eager_execution()
            raise

    def _validate_model(self):
        """Validate loaded model"""
        try:
            # Test with dummy data
            dummy_input = np.random.random((1, *self.target_size, 3))
            prediction = self.model.predict(dummy_input, verbose=0)
            
            if prediction.shape[1] != 15:  # Should have 15 classes
                print(f"❌ Wrong output shape: {prediction.shape}")
                return False
                
            print(f"✅ Model validation successful! Output shape: {prediction.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Model validation failed: {e}")
            return False

    def _load_classes(self):
        """Load class names"""
        if self.classes is not None:
            return

        class_file = os.path.join(os.path.dirname(__file__), "class_names.json")
        
        try:
            with open(class_file, 'r') as f:
                self.classes = json.load(f)
            print(f"✅ Loaded {len(self.classes)} class names: {self.classes}")
        except Exception as e:
            print(f"❌ Could not load class names: {e}")
            # Default class names as fallback
            self.classes = [
                'Alat_Pembersih_Kimia', 'Alumunium', 'Baterai', 'Kaca', 'Kardus',
                'Karet', 'Kertas', 'Lampu_dan_Elektronik', 'Minyak_dan_Oli_Bekas',
                'Obat_dan_Medis', 'Plastik', 'Sisa_Buah_dan_Sayur', 'Sisa_Makanan',
                'Styrofoam', 'Tekstil'
            ]
            print(f"✅ Using default class names: {len(self.classes)} classes")

    def predict(self, image_data):
        """Predict waste category from image data"""
        try:
            print("🔍 Starting classification...")
            
            # Load model and classes
            self._load_model()
            self._load_classes()

            # Preprocess image
            processed_image = self._preprocess_image(image_data)
            print(f"✅ Image preprocessed: {processed_image.shape}")

            # Make prediction
            print("🧠 Running model prediction...")
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get predicted class
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            if predicted_class_idx >= len(self.classes):
                raise ValueError(f"Predicted class index {predicted_class_idx} out of range")
            
            subcategory = self.classes[predicted_class_idx]
            main_category = self.kategori_mapping.get(subcategory, "Unknown")

            print(f"✅ Classification complete:")
            print(f"   Subcategory: {subcategory}")
            print(f"   Main category: {main_category}")
            print(f"   Confidence: {confidence:.3f}")

            return {
                "subcategory": subcategory,
                "main_category": main_category,
                "confidence": confidence,
                "success": True
            }

        except Exception as e:
            error_msg = f"Classification error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "error": error_msg,
                "success": False
            }

    def _preprocess_image(self, image_data):
        """Preprocess image for model input"""
        try:
            # Open image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize
            image = image.resize(self.target_size)

            # Convert to array and normalize
            image_array = np.array(image) / 255.0

            # Add batch dimension
            return np.expand_dims(image_array, axis=0)

        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {str(e)}")

# Function to get classifier instance (for API)
def get_classifier():
    """Get classifier instance"""
    return WasteClassifier()
