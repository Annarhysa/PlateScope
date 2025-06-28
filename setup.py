#!/usr/bin/env python3
"""
Setup script for PlateScope License Plate Recognition System
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing Python dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_fonts():
    """Check if required fonts are present"""
    print("\n🔤 Checking font files...")
    
    required_fonts = [
        'fonts/NotoSansDevanagari-Regular.ttf',
        'fonts/NotoSansGujarati-Regular.ttf',
        'fonts/NotoSansTelugu-Regular.ttf',
        'fonts/NotoSansKannada-Regular.ttf',
        'fonts/NotoSansGurmukhi-Regular.ttf',
        'fonts/NotoSansTamil-Regular.ttf'
    ]
    
    missing_fonts = []
    for font in required_fonts:
        if not os.path.exists(font):
            missing_fonts.append(font)
        else:
            print(f"✅ {font}")
    
    if missing_fonts:
        print(f"\n❌ Missing font files: {len(missing_fonts)}")
        for font in missing_fonts:
            print(f"   - {font}")
        print("\nPlease download the required fonts and place them in the fonts/ directory.")
        return False
    
    print("✅ All required fonts are present!")
    return True

def generate_training_data():
    """Generate license plate images for training"""
    print("\n🖼️ Generating training data...")
    
    try:
        subprocess.check_call([sys.executable, "generate_all_plate_types.py"])
        print("✅ Training data generated successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate training data: {e}")
        return False

def train_models():
    """Train the recognition models"""
    print("\n🤖 Training recognition models...")
    print("This may take 10-30 minutes depending on your hardware.")
    
    try:
        subprocess.check_call([sys.executable, "train_model.py"])
        print("✅ Models trained successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to train models: {e}")
        return False

def test_system():
    """Test the system with a demo"""
    print("\n🧪 Testing system...")
    
    try:
        subprocess.check_call([sys.executable, "demo.py"])
        print("✅ System test completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚗 PlateScope - License Plate Recognition System Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed at dependency installation.")
        return
    
    # Check fonts
    if not check_fonts():
        print("\n⚠️  Setup incomplete due to missing fonts.")
        print("You can continue with the setup, but font-dependent features may not work.")
        response = input("Continue with setup? (y/n): ").lower()
        if response != 'y':
            return
    
    # Generate training data
    if not generate_training_data():
        print("\n❌ Setup failed at training data generation.")
        return
    
    # Train models
    print("\n🤖 Do you want to train the models now? (This may take 10-30 minutes)")
    response = input("Train models? (y/n): ").lower()
    
    if response == 'y':
        if not train_models():
            print("\n❌ Setup failed at model training.")
            return
    
    # Test system
    print("\n🧪 Do you want to test the system?")
    response = input("Run system test? (y/n): ").lower()
    
    if response == 'y':
        test_system()
    
    # Setup complete
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. To start the web application: streamlit run app.py")
    print("2. To run a demo: python demo.py")
    print("3. To train models later: python train_model.py")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user.")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}") 