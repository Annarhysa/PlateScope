import cv2
import numpy as np
import tensorflow as tf
import keras
import easyocr
import os
import json
from PIL import Image
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

class LicensePlateRecognitionModel:
    def __init__(self):
        self.plate_type_model = None
        self.language_model = None
        # Use only English for OCR - we'll handle language detection separately
        self.reader = easyocr.Reader(['en'])
        self.plate_types = ['white', 'yellow', 'green_private', 'green_commercial', 
                           'black', 'red', 'blue', 'military', 'vip']
        self.languages = ['hindi', 'marathi', 'gujarati', 'telugu', 'kannada', 'punjabi', 'tamil']
        
        # Language to script mapping
        self.language_scripts = {
            'hindi': 'Devanagari',
            'marathi': 'Devanagari', 
            'gujarati': 'Gujarati',
            'telugu': 'Telugu',
            'kannada': 'Kannada',
            'punjabi': 'Gurmukhi',
            'tamil': 'Tamil'
        }
        
        # Numeral mappings for conversion
        self.numeral_maps = {
            'hindi': dict(zip('०१२३४५६७८९', '0123456789')),
            'marathi': dict(zip('०१२३४५६७८९', '0123456789')),
            'gujarati': dict(zip('૦૧૨૩૪૫૬૭૮૯', '0123456789')),
            'telugu': dict(zip('౦౧౨౩౪౫౬౭౮౯', '0123456789')),
            'kannada': dict(zip('೦೧೨೩೪೫೬೭೮೯', '0123456789')),
            'punjabi': dict(zip('੦੧੨੩੪੫੬੭੮੯', '0123456789')),
            'tamil': dict(zip('௦௧௨௩௪௫௬௭௮௯', '0123456789'))
        }
        
        # State name mappings
        self.state_names = {
            'hindi': ['उत्तर प्रदेश', 'मध्य प्रदेश', 'दिल्ली', 'राजस्थान', 'हरियाणा'],
            'marathi': ['महाराष्ट्र', 'मुंबई', 'पुणे', 'नागपूर', 'नाशिक'],
            'gujarati': ['ગુજરાત', 'અમદાવાદ', 'સુરત', 'વડોદરા', 'રાજકોટ'],
            'telugu': ['ఆంధ్ర ప్రదేశ్', 'తెలంగాణ', 'హైదరాబాద్', 'విజయవాడ', 'విశాఖపట్నం'],
            'kannada': ['ಕರ್ನಾಟಕ', 'ಬೆಂಗಳೂರು', 'ಮೈಸೂರು', 'ಹುಬ್ಬಳ್ಳಿ', 'ಮಂಗಳೂರು'],
            'punjabi': ['ਪੰਜਾਬ', 'ਚੰਡੀਗੜ੍ਹ', 'ਲੁਧਿਆਣਾ', 'ਅੰਮ੍ਰਿਤਸਰ', 'ਜਲੰਧਰ'],
            'tamil': ['தமிழ்நாடு', 'சென்னை', 'மதுரை', 'கோயம்புத்தூர்', 'திருச்சி']
        }
        
        # Series mappings
        self.series_maps = {
            'hindi': ['बी टी', 'सी ए', 'डी एल', 'पी यू', 'आर जे'],
            'marathi': ['एम ए', 'पी एन', 'एन जी', 'एन एस', 'एम एच'],
            'gujarati': ['જી એ', 'એસ આર', 'એમ ડી', 'વી એ', 'આર જે'],
            'telugu': ['ఏ పి', 'టి ఎస్', 'హెచ్ వై', 'విజె', 'విఎ'],
            'kannada': ['ಕೆ ಎ', 'ಬಿ ಎನ್', 'ಎಂ ಎಸ್', 'ಹು ಬಿ', 'ಮಾ ಎ'],
            'punjabi': ['ਪੀ ਬੀ', 'ਐਲ ਡੀ ਐਚ', 'ਏ ਐਮ ਐਸ', 'ਜਲੰਧਰ', 'ਚੰਡੀਗੜ੍ਹ'],
            'tamil': ['டி என்', 'சி என்', 'எம் எஸ்', 'கோ', 'டி எஸ்']
        }
        
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Preprocess image for model input"""
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        else:
            img = image_path
            
        if img is None:
            return None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img / 255.0  # Normalize
        return img
    
    def extract_features(self, image):
        """Extract color and texture features from image"""
        features = []
        
        # Color features
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        features.extend([
            np.mean(hsv[:, :, 0]),  # Hue mean
            np.std(hsv[:, :, 0]),   # Hue std
            np.mean(hsv[:, :, 1]),  # Saturation mean
            np.std(hsv[:, :, 1]),   # Saturation std
            np.mean(hsv[:, :, 2]),  # Value mean
            np.std(hsv[:, :, 2]),   # Value std
        ])
        
        # Texture features (simple edge density)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # Color histogram features
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [8], [0, 256])
            features.extend(hist.flatten())
        
        return np.array(features)
    
    def build_plate_type_model(self):
        """Build CNN model for plate type classification"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(len(self.plate_types), activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def build_language_model(self):
        """Build CNN model for language classification"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(len(self.languages), activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def load_training_data(self):
        """Load and prepare training data from images directory"""
        X = []
        plate_type_labels = []
        language_labels = []
        
        images_dir = 'images'
        if not os.path.exists(images_dir):
            print(f"Images directory {images_dir} not found!")
            return None, None, None
        
        for filename in os.listdir(images_dir):
            if filename.endswith('.png'):
                # Parse filename: language_platetype_number.png
                parts = filename.replace('.png', '').split('_')
                if len(parts) >= 3:
                    language = parts[0]
                    plate_type = parts[1]
                    
                    if language in self.languages and plate_type in self.plate_types:
                        image_path = os.path.join(images_dir, filename)
                        processed_img = self.preprocess_image(image_path)
                        
                        if processed_img is not None:
                            X.append(processed_img)
                            plate_type_labels.append(self.plate_types.index(plate_type))
                            language_labels.append(self.languages.index(language))
        
        X = np.array(X)
        plate_type_labels = np.array(plate_type_labels)
        language_labels = np.array(language_labels)
        
        return X, plate_type_labels, language_labels
    
    def train_models(self, epochs=10, batch_size=32):
        """Train both plate type and language classification models"""
        print("Loading training data...")
        X, plate_type_labels, language_labels = self.load_training_data()
        
        if X is None or len(X) == 0:
            print("No training data found!")
            return
        
        print(f"Loaded {len(X)} training samples")
        
        # Split data
        X_train, X_test, pt_train, pt_test = train_test_split(
            X, plate_type_labels, test_size=0.2, random_state=42, stratify=plate_type_labels
        )
        _, _, lang_train, lang_test = train_test_split(
            X, language_labels, test_size=0.2, random_state=42, stratify=language_labels
        )
        
        # Train plate type model
        print("Training plate type classification model...")
        self.plate_type_model = self.build_plate_type_model()
        self.plate_type_model.fit(
            X_train, pt_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, pt_test),
            verbose="1"
        )
        
        # Train language model
        print("Training language classification model...")
        self.language_model = self.build_language_model()
        self.language_model.fit(
            X_train, lang_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, lang_test),
            verbose="1"
        )
        
        # Save models
        self.plate_type_model.save('models/plate_type_model.h5')
        self.language_model.save('models/language_model.h5')
        print("Models saved successfully!")
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            os.makedirs('models', exist_ok=True)
            if os.path.exists('models/plate_type_model.h5'):
                self.plate_type_model = keras.models.load_model('models/plate_type_model.h5')
                print("Plate type model loaded successfully!")
            if os.path.exists('models/language_model.h5'):
                self.language_model = keras.models.load_model('models/language_model.h5')
                print("Language model loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def predict_plate_type(self, image):
        """Predict the type of license plate"""
        if self.plate_type_model is None:
            return None
        
        processed_img = self.preprocess_image(image)
        if processed_img is None:
            return None
        
        processed_img = np.expand_dims(processed_img, axis=0)
        prediction = self.plate_type_model.predict(processed_img)  # type: ignore
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        return {
            'type': self.plate_types[predicted_class],
            'confidence': float(confidence)
        }
    
    def predict_language(self, image):
        """Predict the language of the license plate"""
        if self.language_model is None:
            return None
        
        processed_img = self.preprocess_image(image)
        if processed_img is None:
            return None
        
        processed_img = np.expand_dims(processed_img, axis=0)
        prediction = self.language_model.predict(processed_img)  # type: ignore
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        return {
            'language': self.languages[predicted_class],
            'confidence': float(confidence)
        }
    
    def extract_text(self, image):
        """Extract text from license plate using EasyOCR"""
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image
            
        if img is None:
            return []
        
        # Convert to RGB for EasyOCR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Read text with EasyOCR
        results = self.reader.readtext(img_rgb)
        
        extracted_text = []
        combined_text = ""
        
        for (bbox, text, confidence) in results:
            if isinstance(confidence, (int, float)) and confidence > 0.3:  # Filter low confidence results
                extracted_text.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
                combined_text += text + " "
        
        # Add combined text as a separate result
        if combined_text.strip():
            extracted_text.append({
                'text': combined_text.strip(),
                'confidence': np.mean([t['confidence'] for t in extracted_text]) if extracted_text else 0.5,
                'bbox': None
            })
        
        return extracted_text
    
    def convert_to_english_numerals(self, text, language):
        """Convert regional numerals to English numerals"""
        if language in self.numeral_maps:
            numeral_map = self.numeral_maps[language]
            for regional, english in numeral_map.items():
                text = text.replace(regional, english)
        return text
    
    def parse_license_plate_text(self, text, language):
        """Parse license plate text to extract components"""
        # Convert numerals to English
        english_text = self.convert_to_english_numerals(text, language)
        
        # Try to extract state, district, series, and number
        components = {
            'state': '',
            'district': '',
            'series': '',
            'number': '',
            'full_text': text,
            'english_text': english_text
        }
        
        # Clean and split text
        cleaned_text = text.strip()
        words = cleaned_text.split()
        
        if len(words) >= 4:
            # Try to identify components based on common patterns
            # Format: State District Series Number
            if len(words) >= 4:
                # Last 3 parts are usually district, series, number
                components['number'] = words[-1]
                components['series'] = words[-2]
                components['district'] = words[-3]
                components['state'] = ' '.join(words[:-3])
            elif len(words) == 3:
                # Format: State Series Number
                components['number'] = words[-1]
                components['series'] = words[-2]
                components['state'] = words[0]
            elif len(words) == 2:
                # Format: Series Number
                components['series'] = words[0]
                components['number'] = words[1]
        elif len(words) == 1:
            # Single word - might be just the number
            components['number'] = words[0]
        
        return components
    
    def recognize_license_plate(self, image_path):
        """Complete license plate recognition pipeline"""
        results = {
            'plate_type': None,
            'language': None,
            'text_components': None,
            'extracted_text': [],
            'confidence': 0.0
        }
        
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
            else:
                image = image_path
                
            if image is None:
                return results
            
            # Predict plate type
            plate_type_result = self.predict_plate_type(image)
            if plate_type_result:
                results['plate_type'] = plate_type_result
            
            # Predict language
            language_result = self.predict_language(image)
            if language_result:
                results['language'] = language_result
            
            # Extract text
            extracted_text = self.extract_text(image)
            results['extracted_text'] = extracted_text
            
            # Parse text if language is detected
            if language_result and extracted_text:
                # Find the best text result (either combined or highest confidence)
                best_text = None
                for text_result in extracted_text:
                    if text_result['text'] and len(text_result['text'].strip()) > 3:
                        if best_text is None or text_result['confidence'] > best_text['confidence']:
                            best_text = text_result
                
                if best_text:
                    text_components = self.parse_license_plate_text(
                        best_text['text'], 
                        language_result['language']
                    )
                    results['text_components'] = text_components
            
            # Calculate overall confidence
            confidences = []
            if plate_type_result:
                confidences.append(plate_type_result['confidence'])
            if language_result:
                confidences.append(language_result['confidence'])
            if extracted_text:
                confidences.append(max([t['confidence'] for t in extracted_text]))
            
            if confidences:
                results['confidence'] = np.mean(confidences)
            
        except Exception as e:
            print(f"Error in license plate recognition: {e}")
        
        return results

# Utility function to create training data
def create_training_dataset():
    """Create a comprehensive training dataset from the generated images"""
    model = LicensePlateRecognitionModel()
    model.train_models(epochs=15, batch_size=16)
    return model

if __name__ == "__main__":
    # Create and train the model
    print("Creating and training license plate recognition model...")
    model = create_training_dataset()
    print("Training completed!") 