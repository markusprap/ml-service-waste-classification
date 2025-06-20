import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import io

class WasteClassifier:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "model-update.h5")
        print(f"Loading model from: {model_path}")
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
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logs

        try:
            # Try to load model with various compatibility settings
            print("Attempting to load TensorFlow model...")
            
            # Method 1: Standard loading
            try:
                self.model = tf.keras.models.load_model(model_path, compile=False)
                print("✅ Model loaded successfully with standard method!")
            except Exception as e1:
                print(f"❌ Standard loading failed: {e1}")
                
                # Method 2: Load with custom objects for InputLayer compatibility
                try:
                    custom_objects = {
                        'InputLayer': tf.keras.layers.InputLayer,
                        'Dense': tf.keras.layers.Dense,
                        'Conv2D': tf.keras.layers.Conv2D,
                        'MaxPooling2D': tf.keras.layers.MaxPooling2D,
                        'Flatten': tf.keras.layers.Flatten,
                        'Dropout': tf.keras.layers.Dropout
                    }
                    self.model = tf.keras.models.load_model(
                        model_path, 
                        compile=False, 
                        custom_objects=custom_objects
                    )
                    print("✅ Model loaded successfully with custom objects!")
                except Exception as e2:
                    print(f"❌ Custom objects loading failed: {e2}")
                    
                    # Method 3: Try with different TF compatibility
                    try:
                        # Disable eager execution temporarily
                        tf.compat.v1.disable_eager_execution()
                        self.model = tf.keras.models.load_model(model_path, compile=False)
                        tf.compat.v1.enable_eager_execution()
                        print("✅ Model loaded successfully with v1 compatibility!")
                    except Exception as e3:
                        print(f"❌ V1 compatibility loading failed: {e3}")
                        raise Exception(f"All model loading methods failed. Last error: {e3}")
                        
        except Exception as e:
            print(f"💥 CRITICAL ERROR: Cannot load model")
            print(f"Error: {e}")
            print(f"Model path: {model_path}")
            print(f"Model file exists: {os.path.exists(model_path)}")
            print(f"Model file size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
            print(f"TensorFlow version: {tf.__version__}")
            raise e

        # Load class names
        self._load_class_names()
        
        # Validate model
        self._validate_model()

    def _load_class_names(self):
        """Load class names from JSON file"""
        class_names_path = os.path.join(os.path.dirname(__file__), "class_names.json")
        if os.path.exists(class_names_path):
            try:
                with open(class_names_path, "r", encoding="utf-8") as f:
                    self.classes = json.load(f)
                print(f"✅ Loaded {len(self.classes)} class names: {self.classes}")
            except Exception as e:
                print(f"❌ Failed to load class_names.json: {e}")
                self.classes = None

        # Fallback if no class names file
        if not self.classes:
            try:
                # Try to get output shape from model
                if hasattr(self.model, 'output_shape'):
                    if isinstance(self.model.output_shape, tuple):
                        num_classes = self.model.output_shape[-1]
                    else:
                        num_classes = self.model.output_shape[0][-1]
                else:
                    # Get from last layer
                    num_classes = self.model.layers[-1].output_shape[-1]
                
                self.classes = [f'class_{i}' for i in range(num_classes)]
                print(f"⚠️ Using fallback class names for {num_classes} classes")
            except Exception as e:
                print(f"❌ Could not determine number of classes: {e}")
                # Use the known 15 classes from JSON as ultimate fallback
                self.classes = [
                    "Alat_Pembersih_Kimia", "Alumunium", "Baterai", "Kaca", "Kardus",
                    "Karet", "Kertas", "Lampu_dan_Elektronik", "Minyak_dan_Oli_Bekas",
                    "Obat_dan_Medis", "Plastik", "Sisa_Buah_dan_Sayur", "Sisa_Makanan",
                    "Styrofoam", "Tekstil"
                ]
                print(f"🔄 Using hardcoded class names: {len(self.classes)} classes")

    def _validate_model(self):
        """Validate model by testing prediction with dummy data"""
        try:
            print("🧪 Testing model with dummy data...")
            dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
            predictions = self.model.predict(dummy_input, verbose=0)
            print(f"✅ Model validation successful! Output shape: {predictions.shape}")
            print(f"✅ Number of classes: {predictions.shape[-1]}")
        except Exception as e:
            print(f"❌ Model validation failed: {e}")
            raise e

    def preprocess_image(self, image_bytes):
        """Preprocess image for model input"""
        try:
            # Load and convert image
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Resize to model input size
            image = image.resize(self.target_size)
            
            # Convert to array and normalize
            img_array = tf.keras.utils.img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize to [0,1]
            
            print(f"✅ Image preprocessed: {img_array.shape}")
            return img_array
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")

    def predict(self, image_bytes):
        """Predict waste classification"""
        try:
            print("🔍 Starting classification...")
            
            # Preprocess image
            processed_image = self.preprocess_image(image_bytes)
            
            # Make prediction
            print("🧠 Running model prediction...")
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get predicted class
            predicted_class_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class_index])
            
            # Get class name
            if self.classes and len(self.classes) > predicted_class_index:
                predicted_subcategory = self.classes[predicted_class_index]
            else:
                predicted_subcategory = f"class_{predicted_class_index}"
            
            # Get main category
            predicted_main_category = self.kategori_mapping.get(predicted_subcategory, "Tidak diketahui")
            
            print(f"✅ Classification complete:")
            print(f"   Subcategory: {predicted_subcategory}")
            print(f"   Main category: {predicted_main_category}")
            print(f"   Confidence: {confidence:.3f}")
            
            return {
                "subcategory": predicted_subcategory,
                "main_category": predicted_main_category,
                "confidence": confidence,
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Classification error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
