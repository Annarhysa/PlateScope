#!/usr/bin/env python3
"""
Demo script for PlateScope License Plate Recognition System
"""

import os
import cv2
from plate_recognition_model import LicensePlateRecognitionModel

def demo_recognition():
    """Demonstrate license plate recognition functionality"""
    
    print("🚗 PlateScope - License Plate Recognition Demo")
    print("=" * 50)
    
    # Check if models exist
    if not os.path.exists('models/plate_type_model.h5') or not os.path.exists('models/language_model.h5'):
        print("❌ Trained models not found!")
        print("Please run 'python train_model.py' first to train the models.")
        return
    
    # Check if images exist
    if not os.path.exists('images'):
        print("❌ Images directory not found!")
        print("Please run 'python generate_all_plate_types.py' first to generate sample images.")
        return
    
    # Initialize model
    print("🔧 Loading recognition model...")
    model = LicensePlateRecognitionModel()
    model.load_models()
    
    # Get sample images
    image_files = [f for f in os.listdir('images') if f.endswith('.png')]
    
    if len(image_files) == 0:
        print("❌ No sample images found!")
        return
    
    print(f"📊 Found {len(image_files)} sample images")
    
    # Test with first few images
    test_images = image_files[:3]  # Test with first 3 images
    
    for i, image_file in enumerate(test_images, 1):
        print(f"\n🔍 Testing Image {i}: {image_file}")
        print("-" * 40)
        
        image_path = os.path.join('images', image_file)
        
        # Perform recognition
        results = model.recognize_license_plate(image_path)
        
        # Display results
        if results['plate_type']:
            plate_type = results['plate_type']['type']
            plate_confidence = results['plate_type']['confidence']
            print(f"🏷️  Plate Type: {plate_type.replace('_', ' ').title()}")
            print(f"   Confidence: {plate_confidence:.1%}")
        
        if results['language']:
            language = results['language']['language']
            lang_confidence = results['language']['confidence']
            print(f"🌐 Language: {language.title()}")
            print(f"   Confidence: {lang_confidence:.1%}")
        
        if results['text_components']:
            components = results['text_components']
            print(f"📝 Text: {components['full_text']}")
            print(f"🔤 English: {components['english_text']}")
            
            if components.get('state'):
                print(f"🏛️  State: {components['state']}")
            if components.get('district'):
                print(f"📍 District: {components['district']}")
            if components.get('series'):
                print(f"🔢 Series: {components['series']}")
            if components.get('number'):
                print(f"🔢 Number: {components['number']}")
        
        print(f"📊 Overall Confidence: {results.get('confidence', 0):.1%}")
        
        if i < len(test_images):
            print("\n" + "="*50)
    
    print(f"\n✅ Demo completed! Tested {len(test_images)} images.")
    print("\n💡 To use the web interface, run: streamlit run app.py")

def demo_text_extraction():
    """Demonstrate text extraction capabilities"""
    
    print("\n🔤 Text Extraction Demo")
    print("=" * 30)
    
    # Initialize model
    model = LicensePlateRecognitionModel()
    
    # Test numeral conversion
    test_cases = [
        ('hindi', 'उत्तर प्रदेश ५० बी टी ७७३०'),
        ('gujarati', 'ગુજરાત ૪૫ જી એ ૮૭૬૫'),
        ('tamil', 'தமிழ்நாடு ௫௦ டி என் ௭௭௩௦'),
        ('telugu', 'ఆంధ్ర ప్రదేశ్ ౫౧ ఏ పి ౧౨౩౪')
    ]
    
    for language, text in test_cases:
        english_text = model.convert_to_english_numerals(text, language)
        print(f"{language.title()}: {text}")
        print(f"English: {english_text}")
        print()

if __name__ == "__main__":
    try:
        demo_recognition()
        demo_text_extraction()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Please ensure all dependencies are installed and models are trained.") 