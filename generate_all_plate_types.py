from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import random
import math

# Font paths (must be in the font/ directory)
FONTS = {
    'Hindi': 'fonts/NotoSansDevanagari-Regular.ttf',
    'Marathi': 'fonts/NotoSansDevanagari-Regular.ttf',
    'Gujarati': 'fonts/NotoSansGujarati-Regular.ttf',
    'Telugu': 'fonts/NotoSansTelugu-Regular.ttf',
    'Kannada': 'fonts/NotoSansKannada-Regular.ttf',
    'Punjabi': 'fonts/NotoSansGurmukhi-Regular.ttf',
    'Tamil': 'fonts/NotoSansTamil-Regular.ttf',
}

NUMERAL_MAPS = {
    'Hindi': dict(zip('0123456789', '०१२३४५६७८९')),
    'Marathi': dict(zip('0123456789', '०१२३४५६७८९')),
    'Gujarati': dict(zip('0123456789', '૦૧૨૩૪૫૬૭૮૯')),
    'Telugu': dict(zip('0123456789', '౦౧౨౩౪౫౬౭౮౯')),
    'Kannada': dict(zip('0123456789', '೦೧೨೩೪೫೬೭೮೯')),
    'Punjabi': dict(zip('0123456789', '੦੧੨੩੪੫੬੭੮੯')),
    'Tamil': dict(zip('0123456789', '௦௧௨௩௪௫௬௭௮௯')),
}

def to_local_numeral(num_str, lang):
    return ''.join(NUMERAL_MAPS[lang].get(ch, ch) for ch in str(num_str))

STATE_NAMES = {
    'Hindi': ['उत्तर प्रदेश', 'मध्य प्रदेश', 'दिल्ली', 'राजस्थान', 'हरियाणा'],
    'Marathi': ['महाराष्ट्र', 'मुंबई', 'पुणे', 'नागपूर', 'नाशिक'],
    'Gujarati': ['ગુજરાત', 'અમદાવાદ', 'સુરત', 'વડોદરા', 'રાજકોટ'],
    'Telugu': ['ఆంధ్ర ప్రదేశ్', 'తెలంగాణ', 'హైదరాబాద్', 'విజయవాడ', 'విశాఖపట్నం'],
    'Kannada': ['ಕರ್ನಾಟಕ', 'ಬೆಂಗಳೂರು', 'ಮೈಸೂರು', 'ಹುಬ್ಬಳ್ಳಿ', 'ಮಂಗಳೂರು'],
    'Punjabi': ['ਪੰਜਾਬ', 'ਚੰਡੀਗੜ੍ਹ', 'ਲੁਧਿਆਣਾ', 'ਅੰਮ੍ਰਿਤਸਰ', 'ਜਲੰਧਰ'],
    'Tamil': ['தமிழ்நாடு', 'சென்னை', 'மதுரை', 'கோயம்புத்தூர்', 'திருச்சி'],
}

SERIES = {
    'Hindi': ['बी टी', 'सी ए', 'डी एल', 'पी यू', 'आर जे'],
    'Marathi': ['एम ए', 'पी एन', 'एन जी', 'एन एस', 'एम एच'],
    'Gujarati': ['જી એ', 'એસ આર', 'એમ ડી', 'વી એ', 'આર જે'],
    'Telugu': ['ఏ పి', 'టి ఎస్', 'హెచ్ వై', 'విజె', 'విఎ'],
    'Kannada': ['ಕೆ ಎ', 'ಬಿ ಎನ್', 'ಎಂ ಎಸ್', 'ಹು ಬಿ', 'ಮಾ ಎ'],
    'Punjabi': ['ਪੀ ਬੀ', 'ਐਲ ਡੀ ਐਚ', 'ਏ ਐਮ ਐਸ', 'ਜਲੰਧਰ', 'ਚੰਡੀਗੜ੍ਹ'],
    'Tamil': ['டி என்', 'சி என்', 'எம் எஸ்', 'கோ', 'டி எஸ்'],
}

# Plate types and their color schemes
PLATE_TYPES = [
    # (type, bg, fg, extra_fg, description)
    ('white', 'white', 'black', None, 'Private'),
    ('yellow', '#ffe600', 'black', None, 'Commercial'),
    ('green_private', '#1fa055', 'white', None, 'Private EV'),
    ('green_commercial', '#1fa055', 'yellow', None, 'Commercial EV'),
    ('black', 'black', '#ffe600', None, 'Rental'),
    ('red', '#c00', 'white', None, 'Temporary'),
    ('blue', '#1e4db7', 'white', None, 'Diplomatic'),
    ('military', '#ffe600', 'black', None, 'Military'),
    ('vip', '#c00', 'white', None, 'VIP'),
]

WIDTH, HEIGHT = 420, 110
DOT_RADIUS = 5
DOT_Y = 18 + 10
FONT_SIZE = 32

os.makedirs('images', exist_ok=True)

# Helper for camera effect
def camera_effect(img):
    # Slight rotation
    angle = random.uniform(-2, 2)
    img = img.rotate(angle, expand=1, fillcolor='white')
    # Slight blur
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    # Add noise
    pixels = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if random.random() < 0.01:
                r, g, b = pixels[i, j]
                noise = random.randint(-20, 20)
                pixels[i, j] = (max(0, min(255, r+noise)), max(0, min(255, g+noise)), max(0, min(255, b+noise)))
    return img

# Helper for military arrow
def draw_military_arrow(draw, x, y, size=18):
    # Draw a simple upward arrow
    draw.polygon([
        (x, y+size), (x+size//2, y), (x+size, y+size), (x+size//2, y+size//2)
    ], fill='black')

# Helper for VIP emblem
def draw_vip_emblem(draw, x, y, size=24):
    # Draw a simple Ashoka Chakra-like circle
    draw.ellipse([(x, y), (x+size, y+size)], outline='white', width=2)
    draw.line([(x+size//2, y+4), (x+size//2, y+size-4)], fill='white', width=2)
    draw.line([(x+4, y+size//2), (x+size-4, y+size//2)], fill='white', width=2)

for lang in FONTS:
    font_path = FONTS[lang]
    font = ImageFont.truetype(font_path, FONT_SIZE)
    for plate_type, bg, fg, extra_fg, desc in PLATE_TYPES:
        for i in range(10):
            state = random.choice(STATE_NAMES[lang])
            district = to_local_numeral(random.randint(10, 99), lang)
            series = random.choice(SERIES[lang])
            number = to_local_numeral(random.randint(1000, 9999), lang)
            plate_text = f"{state} {district} {series} {number}"
            # Create image
            img = Image.new('RGB', (WIDTH, HEIGHT), color=bg)
            draw = ImageDraw.Draw(img)
            # Red border at the top for red/vip/temporary
            if plate_type in ['white', 'yellow', 'green_private', 'green_commercial', 'black', 'blue', 'military']:
                draw.rectangle([(0, 0), (WIDTH, 18)], fill=None)
            else:
                draw.rectangle([(0, 0), (WIDTH, 18)], fill='#a00')
            # Black border
            draw.rectangle([(0, 0), (WIDTH-1, HEIGHT-1)], outline='black', width=2)
            # Two mounting dots
            for dot_x in [WIDTH//4, 3*WIDTH//4]:
                draw.ellipse([(dot_x-DOT_RADIUS, DOT_Y-DOT_RADIUS), (dot_x+DOT_RADIUS, DOT_Y+DOT_RADIUS)], fill='black')
            # Center the text
            bbox = draw.textbbox((0, 0), plate_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (WIDTH - text_width) // 2
            y = (HEIGHT - text_height) // 2 + 10
            draw.text((x, y), plate_text, fill=fg, font=font)
            # Special features
            if plate_type == 'military':
                draw_military_arrow(draw, WIDTH-40, 25)
            if plate_type == 'vip':
                draw_vip_emblem(draw, 20, 20)
            # Camera effect
            img = camera_effect(img)
            # Save
            fname = f"images/{lang.lower()}_{plate_type}_{i+1:02d}.png"
            img.save(fname)
            print(f"Generated {fname}") 