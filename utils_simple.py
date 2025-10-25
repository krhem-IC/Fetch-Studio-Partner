from PIL import Image, ImageStat, ImageOps, ImageFilter, ImageDraw
import io
import pytesseract
import imghdr
from config import IMAGE_TYPES, APPROVED_HEX
import random

# register HEIC/HEIF support if available
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

# open any uploaded image and normalize it to a PIL.Image in RGB
def load_uploaded_image(file_obj):
    # Streamlit uploads give a file-like object
    raw = file_obj.read()
    bio = io.BytesIO(raw)

    # try Pillow first
    try:
        img = Image.open(bio)
    except Exception as e:
        raise ValueError(f"Could not open image. Error: {e}")

    # if animated gif or webp, grab first frame
    try:
        if getattr(img, "is_animated", False):
            img.seek(0)
    except Exception:
        pass

    # fix EXIF orientation
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # convert to RGB for consistent processing
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if img.mode == "P" else "RGB")
    if img.mode == "RGBA":
        # flatten any transparency onto white to avoid hidden overlays
        bg = Image.new("RGB", img.size, "#FFFFFF")
        bg.paste(img, mask=img.split()[-1])
        img = bg

    return img


def remove_background(img):
    """
    Remove white/light backgrounds from product images using simple color thresholding.
    Much faster than AI removal and preserves product packaging better.
    """
    try:
        # Convert to RGBA
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Get pixel data
        data = img.getdata()
        
        # Create new image with transparency
        new_data = []
        
        # Define white/light background threshold
        threshold = 240
        
        for item in data:
            r, g, b = item[0], item[1], item[2]
            
            # If pixel is very light (likely background), make it transparent
            if r > threshold and g > threshold and b > threshold:
                new_data.append((r, g, b, 0))  # Transparent
            else:
                # Keep the pixel as-is
                new_data.append((r, g, b, 255))  # Opaque
        
        # Apply the new data
        img.putdata(new_data)
        
        return img
        
    except Exception as e:
        print(f"⚠️  Background removal failed: {e}. Using original image.")
        if img.mode != 'RGBA':
            return img.convert('RGBA')
        return img


def add_drop_shadow(img, offset=(6, 6), blur_radius=10, shadow_color=(0, 0, 0, 80)):
    """Simple drop shadow effect"""
    # Create shadow
    shadow = Image.new('RGBA', (img.width + offset[0] * 2, img.height + offset[1] * 2), (0, 0, 0, 0))
    shadow_mask = Image.new('RGBA', img.size, shadow_color)
    shadow.paste(shadow_mask, offset, img.split()[3] if img.mode == 'RGBA' else None)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # Composite image over shadow
    shadow.paste(img, (0, 0), img.split()[3] if img.mode == 'RGBA' else None)
    return shadow


def create_composite_image(products_rgba, bg_color, target_width, target_height, product_callouts=""):
    """
    SIMPLIFIED, FASTER composition - matches reference images exactly.
    """
    print(f"\n🎨 Creating composition with {len(products_rgba)} products...")
    
    # Create base canvas
    composite = Image.new("RGBA", (target_width, target_height), bg_color + (255,))
    
    num_products = len(products_rgba)
    
    # Simple sizing based on count
    if num_products == 1:
        scale = 0.6
    elif num_products == 2:
        scale = 0.45
    elif num_products == 3:
        scale = 0.35
    else:
        scale = 0.3
    
    # Process all products
    processed = []
    for idx, (img_rgba, filename) in enumerate(products_rgba):
        # Maintain aspect ratio
        aspect = img_rgba.width / img_rgba.height
        target_h = int(target_height * scale)
        target_w = int(target_h * aspect)
        
        # Limit width
        max_w = int(target_width * scale * 1.3)
        if target_w > max_w:
            target_w = max_w
            target_h = int(target_w / aspect)
        
        resized = img_rgba.resize((target_w, target_h), Image.Resampling.LANCZOS)
        with_shadow = add_drop_shadow(resized, offset=(6, 6), blur_radius=10)
        processed.append(with_shadow)
    
    # LAYOUT - Simple and predictable
    if num_products == 1:
        img = processed[0]
        x = (target_width - img.width) // 2
        y = (target_height - img.height) // 2
        composite.paste(img, (x, y), img)
    
    elif num_products == 2:
        gap = int(target_width * 0.06)
        total_w = processed[0].width + gap + processed[1].width
        start_x = (target_width - total_w) // 2
        
        for i, img in enumerate(processed):
            x = start_x if i == 0 else start_x + processed[0].width + gap
            y = (target_height - img.height) // 2
            composite.paste(img, (x, y), img)
    
    elif num_products == 3:
        gap = int(target_width * 0.06)
        total_w = sum(img.width for img in processed) + (2 * gap)
        start_x = (target_width - total_w) // 2
        
        x = start_x
        for img in processed:
            y = (target_height - img.height) // 2
            composite.paste(img, (x, y), img)
            x += img.width + gap
    
    else:  # 4+ products
        if num_products == 4:
            # 2x2 grid
            gap = int(target_width * 0.06)
            
            # Top row
            top_w = processed[0].width + gap + processed[1].width
            top_x = (target_width - top_w) // 2
            
            # Bottom row
            bot_w = processed[2].width + gap + processed[3].width
            bot_x = (target_width - bot_w) // 2
            
            # Heights
            top_h = max(processed[0].height, processed[1].height)
            bot_h = max(processed[2].height, processed[3].height)
            total_h = top_h + gap + bot_h
            start_y = (target_height - total_h) // 2
            
            # Place
            composite.paste(processed[0], (top_x, start_y), processed[0])
            composite.paste(processed[1], (top_x + processed[0].width + gap, start_y), processed[1])
            composite.paste(processed[2], (bot_x, start_y + top_h + gap), processed[2])
            composite.paste(processed[3], (bot_x + processed[2].width + gap, start_y + top_h + gap), processed[3])
        
        else:
            # 5+ products: multi-row
            gap = int(target_width * 0.05)
            margin = int(target_width * 0.1)
            max_row_w = target_width - (2 * margin)
            
            rows = []
            current_row = []
            current_w = 0
            
            for img in processed:
                if current_w + img.width <= max_row_w or not current_row:
                    current_row.append(img)
                    current_w += img.width + (gap if current_w > 0 else 0)
                else:
                    rows.append(current_row)
                    current_row = [img]
                    current_w = img.width
            
            if current_row:
                rows.append(current_row)
            
            # Calculate heights
            row_heights = [max(img.height for img in row) for row in rows]
            total_h = sum(row_heights) + (gap * (len(rows) - 1))
            y = (target_height - total_h) // 2
            
            # Place rows
            for row_idx, row in enumerate(rows):
                row_w = sum(img.width for img in row) + (gap * (len(row) - 1))
                x = (target_width - row_w) // 2
                
                for img in row:
                    composite.paste(img, (x, y), img)
                    x += img.width + gap
                
                y += row_heights[row_idx] + gap
    
    # Convert to RGB
    final = Image.new("RGB", (target_width, target_height), bg_color)
    final.paste(composite, (0, 0), composite)
    
    return final


def parse_product_instructions(callouts_text, filenames):
    """Dummy parser - returns defaults"""
    return [{'size': 'medium', 'position': 'auto', 'priority': 1} for _ in filenames]


def enforce_specs(img, image_type):
    """Apply Fetch brand specifications"""
    spec = IMAGE_TYPES.get(image_type, IMAGE_TYPES["offer_tile"])
    
    # Resize if needed
    target_size = (spec["width"], spec["height"])
    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to correct format
    if spec["format"].upper() == "PNG":
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    else:  # JPG
        if img.mode != "RGB":
            img = img.convert("RGB")
    
    return img


def build_prompt(preset_key, brand, products, background_hex, lifestyle_keywords=""):
    """Build AI prompt (not used in simple mode)"""
    return ""


def validate_brand_compliance(img):
    """Check if image meets Fetch brand guidelines"""
    issues = []
    
    # Check resolution
    if img.width < 1000 or img.height < 800:
        issues.append("Resolution too low (min 1000x800)")
    
    # Check aspect ratio roughly
    aspect = img.width / img.height
    if aspect < 0.7 or aspect > 1.5:
        issues.append(f"Unusual aspect ratio: {aspect:.2f}")
    
    return issues
