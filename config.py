"""
Configuration file for PlateScope License Plate Recognition System
"""

import os

# Model Configuration
MODEL_CONFIG = {
    'plate_type_model_path': 'models/plate_type_model.h5',
    'language_model_path': 'models/language_model.h5',
    'input_size': (224, 224),
    'batch_size': 16,
    'epochs': 15,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'random_seed': 42
}

# OCR Configuration
OCR_CONFIG = {
    'languages': ['en', 'hi', 'gu', 'te', 'kn', 'pa', 'ta', 'mr'],
    'confidence_threshold': 0.3,
    'gpu': True,  # Set to False if GPU is not available
    'model_storage_directory': '~/.EasyOCR'
}

# Image Processing Configuration
IMAGE_CONFIG = {
    'target_size': (224, 224),
    'normalization_factor': 255.0,
    'camera_effect': {
        'rotation_range': (-2, 2),
        'blur_range': (0.3, 1.0),
        'noise_probability': 0.01,
        'noise_range': (-20, 20)
    }
}

# License Plate Types
PLATE_TYPES = {
    'white': {
        'name': 'Private',
        'background': 'white',
        'text_color': 'black',
        'description': 'Private vehicles'
    },
    'yellow': {
        'name': 'Commercial',
        'background': '#ffe600',
        'text_color': 'black',
        'description': 'Commercial vehicles'
    },
    'green_private': {
        'name': 'Private EV',
        'background': '#1fa055',
        'text_color': 'white',
        'description': 'Private electric vehicles'
    },
    'green_commercial': {
        'name': 'Commercial EV',
        'background': '#1fa055',
        'text_color': '#ffe600',
        'description': 'Commercial electric vehicles'
    },
    'black': {
        'name': 'Rental',
        'background': 'black',
        'text_color': '#ffe600',
        'description': 'Rental vehicles'
    },
    'red': {
        'name': 'Temporary',
        'background': '#c00',
        'text_color': 'white',
        'description': 'Temporary/transit vehicles'
    },
    'blue': {
        'name': 'Diplomatic',
        'background': '#1e4db7',
        'text_color': 'white',
        'description': 'Diplomatic vehicles'
    },
    'military': {
        'name': 'Military',
        'background': '#ffe600',
        'text_color': 'black',
        'description': 'Military vehicles'
    },
    'vip': {
        'name': 'VIP',
        'background': '#c00',
        'text_color': 'white',
        'description': 'VIP vehicles'
    }
}

# Language Configuration
LANGUAGES = {
    'hindi': {
        'name': 'हिंदी (Hindi)',
        'script': 'Devanagari',
        'font': 'fonts/NotoSansDevanagari-Regular.ttf',
        'numerals': dict(zip('0123456789', '०१२३४५६७८९')),
        'states': ['उत्तर प्रदेश', 'मध्य प्रदेश', 'दिल्ली', 'राजस्थान', 'हरियाणा'],
        'series': ['बी टी', 'सी ए', 'डी एल', 'पी यू', 'आर जे']
    },
    'marathi': {
        'name': 'मराठी (Marathi)',
        'script': 'Devanagari',
        'font': 'fonts/NotoSansDevanagari-Regular.ttf',
        'numerals': dict(zip('0123456789', '०१२३४५६७८९')),
        'states': ['महाराष्ट्र', 'मुंबई', 'पुणे', 'नागपूर', 'नाशिक'],
        'series': ['एम ए', 'पी एन', 'एन जी', 'एन एस', 'एम एच']
    },
    'gujarati': {
        'name': 'ગુજરાતી (Gujarati)',
        'script': 'Gujarati',
        'font': 'fonts/NotoSansGujarati-Regular.ttf',
        'numerals': dict(zip('0123456789', '૦૧૨૩૪૫૬૭૮૯')),
        'states': ['ગુજરાત', 'અમદાવાદ', 'સુરત', 'વડોદરા', 'રાજકોટ'],
        'series': ['જી એ', 'એસ આર', 'એમ ડી', 'વી એ', 'આર જે']
    },
    'telugu': {
        'name': 'తెలుగు (Telugu)',
        'script': 'Telugu',
        'font': 'fonts/NotoSansTelugu-Regular.ttf',
        'numerals': dict(zip('0123456789', '౦౧౨౩౪౫౬౭౮౯')),
        'states': ['ఆంధ్ర ప్రదేశ్', 'తెలంగాణ', 'హైదరాబాద్', 'విజయవాడ', 'విశాఖపట్నం'],
        'series': ['ఏ పి', 'టి ఎస్', 'హెచ్ వై', 'విజె', 'విఎ']
    },
    'kannada': {
        'name': 'ಕನ್ನಡ (Kannada)',
        'script': 'Kannada',
        'font': 'fonts/NotoSansKannada-Regular.ttf',
        'numerals': dict(zip('0123456789', '೦೧೨೩೪೫೬೭೮೯')),
        'states': ['ಕರ್ನಾಟಕ', 'ಬೆಂಗಳೂರು', 'ಮೈಸೂರು', 'ಹುಬ್ಬಳ್ಳಿ', 'ಮಂಗಳೂರು'],
        'series': ['ಕೆ ಎ', 'ಬಿ ಎನ್', 'ಎಂ ಎಸ್', 'ಹು ಬಿ', 'ಮಾ ಎ']
    },
    'punjabi': {
        'name': 'ਪੰਜਾਬੀ (Punjabi)',
        'script': 'Gurmukhi',
        'font': 'fonts/NotoSansGurmukhi-Regular.ttf',
        'numerals': dict(zip('0123456789', '੦੧੨੩੪੫੬੭੮੯')),
        'states': ['ਪੰਜਾਬ', 'ਚੰਡੀਗੜ੍ਹ', 'ਲੁਧਿਆਣਾ', 'ਅੰਮ੍ਰਿਤਸਰ', 'ਜਲੰਧਰ'],
        'series': ['ਪੀ ਬੀ', 'ਐਲ ਡੀ ਐਚ', 'ਏ ਐਮ ਐਸ', 'ਜਲੰਧਰ', 'ਚੰਡੀਗੜ੍ਹ']
    },
    'tamil': {
        'name': 'தமிழ் (Tamil)',
        'script': 'Tamil',
        'font': 'fonts/NotoSansTamil-Regular.ttf',
        'numerals': dict(zip('0123456789', '௦௧௨௩௪௫௬௭௮௯')),
        'states': ['தமிழ்நாடு', 'சென்னை', 'மதுரை', 'கோயம்புத்தூர்', 'திருச்சி'],
        'series': ['டி என்', 'சி என்', 'எம் எஸ்', 'கோ', 'டி எஸ்']
    }
}

# UI Configuration
UI_CONFIG = {
    'page_title': 'PlateScope - License Plate Recognition',
    'page_icon': '🚗',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'color_palette': {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#4ecdc4',
        'warning': '#ff6b6b',
        'info': '#45b7d1',
        'light': '#f8f9fa',
        'dark': '#343a40'
    },
    'gradients': {
        'primary': 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
        'secondary': 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
        'success': 'linear-gradient(90deg, #ff6b6b, #4ecdc4)'
    }
}

# File Paths
PATHS = {
    'images_dir': 'images',
    'fonts_dir': 'fonts',
    'models_dir': 'models',
    'logs_dir': 'logs',
    'temp_dir': 'temp'
}

# Create necessary directories
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'logs/platescope.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'max_image_size': (1920, 1080),
    'min_confidence': 0.5,
    'max_processing_time': 30,  # seconds
    'cache_results': True,
    'enable_gpu': True,
    'thread_count': 4
}

# Security Configuration
SECURITY_CONFIG = {
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'allowed_extensions': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'],
    'enable_file_validation': True,
    'sanitize_filenames': True
} 