#!/usr/bin/env python3
"""
Training script for the License Plate Recognition Model
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plate_recognition_model import LicensePlateRecognitionModel

def main():
    print("🚗 PlateScope - License Plate Recognition Model Training")
    print("=" * 60)
    
    # Check if images directory exists
    if not os.path.exists('images'):
        print("❌ Error: 'images' directory not found!")
        print("Please run 'python generate_all_plate_types.py' first to generate training data.")
        return
    
    # Count available images
    image_files = [f for f in os.listdir('images') if f.endswith('.png')]
    print(f"📊 Found {len(image_files)} training images")
    
    if len(image_files) == 0:
        print("❌ No training images found!")
        return
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize model
    print("\n🔧 Initializing License Plate Recognition Model...")
    model = LicensePlateRecognitionModel()
    
    # Train models
    print("\n🎯 Starting model training...")
    print("This may take several minutes depending on your hardware.")
    
    try:
        model.train_models(epochs=15, batch_size=16)
        print("\n✅ Training completed successfully!")
        
        # Test the model with a sample image
        print("\n🧪 Testing model with sample image...")
        test_image_path = os.path.join('images', image_files[0])
        
        if os.path.exists(test_image_path):
            results = model.recognize_license_plate(test_image_path)
            
            print(f"\n📋 Sample Test Results:")
            print(f"   Plate Type: {results.get('plate_type', {}).get('type', 'N/A')}")
            print(f"   Language: {results.get('language', {}).get('language', 'N/A')}")
            print(f"   Confidence: {results.get('confidence', 0):.1%}")
            
            if results.get('text_components'):
                components = results['text_components']
                print(f"   Text: {components.get('full_text', 'N/A')}")
                print(f"   English: {components.get('english_text', 'N/A')}")
        
        print("\n🎉 Model is ready for use!")
        print("Run 'streamlit run app.py' to start the web interface.")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("Please check your dependencies and try again.")
        return

if __name__ == "__main__":
    main() 