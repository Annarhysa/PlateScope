# PlateScope

A tool for generating realistic Indian license plate images in various regional languages and scripts, supporting all major Indian number plate types.

## Generated License Plates

This project now generates 10 realistic license plate images for each of the following:
- **Languages:** Hindi, Marathi, Gujarati, Telugu, Kannada, Punjabi, Tamil
- **Plate Types:**
  - White (private)
  - Yellow (commercial)
  - Green (private EV)
  - Green (commercial EV)
  - Black (rental)
  - Red (temporary)
  - Blue (diplomatic)
  - Military
  - Red with Emblem (VIP)

All plates use the correct script, numerals, and a realistic style matching actual Indian license plates, with camera/photo effects for realism.

### File Structure
```
images/
├── hindi_white_01.png ... hindi_vip_10.png
├── marathi_white_01.png ... marathi_vip_10.png
├── gujarati_white_01.png ... gujarati_vip_10.png
├── telugu_white_01.png ... telugu_vip_10.png
├── kannada_white_01.png ... kannada_vip_10.png
├── punjabi_white_01.png ... punjabi_vip_10.png
└── tamil_white_01.png ... tamil_vip_10.png
```

### Plate Style
- Correct background and text color for each type
- White, yellow, green, black, red, blue backgrounds as per Indian law
- Special features: military arrow, VIP emblem, etc.
- Centered, bold text in the local script
- All numbers are in the local script's numerals
- Camera/photo effect: slight rotation, blur, noise
- Font: Noto Sans (from font/ directory) for each script

### Example Plate Text
- **Hindi**: उत्तर प्रदेश ५० बी टी ७७३०
- **Marathi**: महाराष्ट्र ५५ एम ए १२३४
- **Gujarati**: ગુજરાત ૪૫ જી એ ૮૭૬૫
- **Telugu**: ఆంధ్ర ప్రదేశ్ ౫౧ ఏ పి ౧౨౩౪
- **Kannada**: ಕರ್ನಾಟಕ ೫೬ ಕೆ ಎ ೩೪೫೬
- **Punjabi**: ਪੰਜਾਬ ੫੦ ਪੀ ਬੀ ੭੭੩੦
- **Tamil**: தமிழ்நாடு ௫௦ டி என் ௭௭௩௦

## Usage

### Generate All Plate Types
```bash
python generate_all_plate_types.py
```
This will create 10 plates for each language and plate type in the `images/` directory.

## Requirements

- Pillow (PIL) >= 9.0.0
- The following font files must be present in your `font/` directory:
  - NotoSansDevanagari-Regular.ttf
  - NotoSansGujarati-Regular.ttf
  - NotoSansTelugu-Regular.ttf
  - NotoSansKannada-Regular.ttf
  - NotoSansGurmukhi-Regular.ttf
  - NotoSansTamil-Regular.ttf

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Troubleshooting Font Issues
- The `font/` directory must be in your project root.
- Font filenames must match exactly (case-sensitive).
- Each font file should be several hundred KB in size.
- If you get `OSError: cannot open resource`, check the font path and file existence.
- You can print the absolute path in your script for debugging:
  ```python
  import os
  print(os.path.abspath(font_path))
  ```

## Features
- Realistic design and layout for all Indian number plate types
- Correct script and numerals for each language
- High-quality PNG images
- Camera/photo effect for realism
- Easily extensible for more languages or styles

---
If you want to add more languages, plate types, or customize the style, just ask!
