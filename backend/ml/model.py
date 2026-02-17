import numpy as np
from PIL import Image
from io import BytesIO
import os
import logging
import keras
from pathlib import Path

logger = logging.getLogger(__name__)

class PlantDiseaseModel:
    def __init__(self):
        """Initialize the plant disease detection model with the actual .keras model"""
        self.disease_classes = [
            "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
            "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)__Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
            "Corn_(maize)___healthy", "Grape___Black_rot", "Grape__Esca_(Black_Measles)_", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
            "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
            "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
            "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch",
            "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
            "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
        ]
        self.input_size = (224, 224)
        
        model_path = Path(__file__).parent / "assets" / "plant_disease_recog_model_pwp.keras"
        try:
            if model_path.exists():
                self.model = keras.models.load_model(model_path)
                logger.info(f"Model loaded successfully from {model_path}")
            else:
                self.model = None
                logger.error(f"Model file NOT found at {model_path}")
        except Exception as e:
            self.model = None
            logger.error(f"Error loading model: {e}")
            
        logger.info(f"Model initialized with {len(self.disease_classes)} disease classes")
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image for model inference.
        Handles image loading, resizing, and normalization.
        """
        try:
            image = Image.open(BytesIO(image_bytes))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image = image.resize(self.input_size, Image.Resampling.LANCZOS)
            img_array = np.array(image, dtype=np.float32)
            # Normalization might depend on how the model was trained, 
            # but 1/255 is standard for most Keras image models.
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {e}")
    
    def predict(self, image_bytes: bytes) -> dict:
        """
        Make prediction on uploaded image using the actual Keras model.
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded correctly")
                
            processed_image = self.preprocess_image(image_bytes)
            
            predictions = self.model.predict(processed_image, verbose=0)
            
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.disease_classes[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            all_predictions = [
                {
                    "class": self.disease_classes[i],
                    "confidence": round(float(predictions[0][i]), 4)
                }
                for i in range(len(self.disease_classes))
            ]
            all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
            
            return {
                "predicted_disease": predicted_class,
                "confidence": confidence,
                "all_predictions": all_predictions[:5],
                "success": True
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "error": str(e),
                "success": False
            }

plant_disease_model = None

def get_model() -> PlantDiseaseModel:
    """Dependency for obtaining model instance"""
    global plant_disease_model
    if plant_disease_model is None:
        plant_disease_model = PlantDiseaseModel()
    return plant_disease_model