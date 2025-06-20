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
        self.using_fallback = False
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

        try:
            # Set TensorFlow to use CPU only in production to avoid GPU memory issues
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            # Try multiple loading strategies for compatibility
            try:
                # First attempt: Standard loading
                self.model = tf.keras.models.load_model(model_path, compile=False)
                print("Successfully loaded the model!")
            except Exception as e1:
                print(f"Standard loading failed: {e1}")
                print("Trying compatibility mode...")
                
                try:
                    # Second attempt: With safe_mode=False
                    self.model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
                    print("Successfully loaded the model with safe_mode=False!")
                except Exception as e2:
                    print(f"Safe mode failed: {e2}")
                    print("Trying with TensorFlow SavedModel format...")
                    
                    # Third attempt: Force SavedModel loading
                    try:
                        import tensorflow.keras.utils as utils
                        # Try to load as SavedModel if H5 fails
                        self.model = tf.saved_model.load(model_path.replace('.h5', ''))
                        print("Loaded as SavedModel!")
                    except Exception as e3:
                        print(f"SavedModel loading failed: {e3}")
                        
                        # Fourth attempt: Load weights only approach
                        print("Attempting to recreate model architecture...")
                        self.model = self._create_fallback_model()
                        try:
                            self.model.load_weights(model_path)
                            print("Successfully loaded model weights into fallback architecture!")
                        except Exception as e4:
                            print(f"Weight loading failed: {e4}")
                            raise Exception(f"All model loading attempts failed. Last error: {e4}")
                    
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Model path: {model_path}")
            print(f"Model file exists: {os.path.exists(model_path)}")
            print(f"TensorFlow version: {tf.__version__}")
            raise e

        # Load class names
        class_names_path = os.path.join(os.path.dirname(__file__), "class_names.json")
        if os.path.exists(class_names_path):
            try:
                with open(class_names_path, "r", encoding="utf-8") as f:
                    self.classes = json.load(f)
                print(f"Loaded class names from class_names.json: {self.classes}")
            except Exception as e:
                print(f"Failed to load class_names.json: {e}")
                self.classes = None

        if not self.classes:
            if hasattr(self.model, 'class_names'):
                self.classes = list(self.model.class_names)
                print(f"Loaded class names from model.class_names: {self.classes}")
            elif hasattr(self.model, 'classes_'):
                self.classes = list(self.model.classes_)
                print(f"Loaded class names from model.classes_: {self.classes}")
            else:
                try:
                    config = self.model.get_config()
                    if 'layers' in config:
                        for layer in reversed(config['layers']):
                            if 'class_names' in layer.get('config', {}):
                                self.classes = layer['config']['class_names']
                                print(f"Loaded class names from model config: {self.classes}")
                                break
                except Exception as e:
                    print(f"Could not get class names from model config: {e}")

        if not self.classes:
            try:
                if hasattr(self.model, 'output_shape'):
                    num_classes = self.model.output_shape[-1]
                else:
                    num_classes = self.model.layers[-1].output_shape[-1]
                self.classes = [f'class_{i}' for i in range(num_classes)]
                print(f"Class names not found in model, fallback to generic: {self.classes}")
            except Exception as e:
                print(f"Could not determine number of classes: {e}")
                self.classes = []

    def _create_fallback_model(self):
        """Create a fallback model architecture compatible with the weights"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(15, activation='softmax')  # 15 classes
            ])
            print("Created fallback model architecture")
            return model
        except Exception as e:
            print(f"Failed to create fallback model: {e}")
            raise e

    def preprocess_image(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = image.resize(self.target_size)
            img_array = tf.keras.utils.img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            return img_array
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")

    def predict(self, image_bytes):
        try:
            processed_image = self.preprocess_image(image_bytes)
            predictions = self.model.predict(processed_image)
            predicted_class_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class_index])
            predicted_subcategory = self.classes[predicted_class_index] if self.classes and len(self.classes) > predicted_class_index else str(predicted_class_index)
            predicted_main_category = self.kategori_mapping.get(predicted_subcategory, "Tidak diketahui")
            return {
                "subcategory": predicted_subcategory,
                "main_category": predicted_main_category,
                "confidence": confidence,
                "success": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
