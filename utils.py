from PIL import Image, ImageStat, ImageOps, ImageFilter, ImageDraw
import io
import pytesseract
import imghdr
from config import IMAGE_TYPES, APPROVED_HEX
from rembg import remove as rembg_remove

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
    Remove background from product image using AI-powered background removal.
    Returns image with transparent background (RGBA mode).
    
    Args:
        img: PIL Image in RGB mode
    
    Returns:
        PIL.Image: Image with transparent background (RGBA mode)
    """
    try:
        # Convert to bytes for rembg
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Remove background
        output = rembg_remove(img_byte_arr)
        
        # Convert back to PIL Image
        result = Image.open(io.BytesIO(output))
        
        # Ensure RGBA mode
        if result.mode != 'RGBA':
            result = result.convert('RGBA')
        
        return result
    except Exception as e:
        print(f"⚠️  Background removal failed: {e}. Using original image.")
        # Return original image with alpha channel
        if img.mode != 'RGBA':
            return img.convert('RGBA')
        return img


def add_drop_shadow(img, offset=(10, 10), shadow_color=(0, 0, 0, 80), blur_radius=15):
    """
    Add a subtle drop shadow to a product image for depth and realism.
    
    Args:
        img: PIL Image with transparent background (RGBA)
        offset: (x, y) offset for shadow
        shadow_color: RGBA color for shadow
        blur_radius: Blur radius for shadow softness
    
    Returns:
        PIL.Image: Image with drop shadow (RGBA)
    """
    # Create shadow layer
    shadow = Image.new('RGBA', (img.width + abs(offset[0]) + blur_radius*2,
                                 img.height + abs(offset[1]) + blur_radius*2),
                       (0, 0, 0, 0))
    
    # Position for shadow
    shadow_x = blur_radius + max(0, offset[0])
    shadow_y = blur_radius + max(0, offset[1])
    
    # Create shadow from alpha channel
    alpha = img.split()[3]
    shadow_mask = Image.new('RGBA', img.size, shadow_color)
    shadow.paste(shadow_mask, (shadow_x, shadow_y), mask=alpha)
    
    # Blur the shadow
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # Position for original image
    img_x = blur_radius + max(0, -offset[0])
    img_y = blur_radius + max(0, -offset[1])
    
    # Composite original image over shadow
    result = Image.new('RGBA', shadow.size, (0, 0, 0, 0))
    result.paste(shadow, (0, 0))
    result.paste(img, (img_x, img_y), img)
    
    return result


def suggest_filename(preset_key, background_hex, original_name="upload"):
    ext = "png" if IMAGE_TYPES[preset_key]["format"] == "png" else "jpg"
    safe_base = original_name.rsplit(".", 1)[0].replace(" ", "_")
    return f"{safe_base}_{preset_key}_{background_hex[1:]}.{ext}"

def validate_aspect_ratio_preservation(original_img, resized_img, tolerance=0.01):
    """
    Validate that resizing preserved the original aspect ratio within tolerance.
    This is a non-blocking check that logs warnings but doesn't prevent processing.
    
    Args:
        original_img: Original PIL Image
        resized_img: Resized PIL Image  
        tolerance: Acceptable deviation (default 1%)
    
    Returns:
        bool: True if aspect ratio is preserved within tolerance
    """
    original_aspect = original_img.width / original_img.height
    resized_aspect = resized_img.width / resized_img.height
    
    deviation = abs(original_aspect - resized_aspect) / original_aspect
    return deviation <= tolerance


def create_composite_image(loaded_images, image_type, background_hex, product_callouts=""):
    """
    Create a Fetch-style composite image with natural product arrangement.
    
    Products are automatically background-removed and composed with:
    - Natural, varied positioning (not rigid grids)
    - Size variation for visual hierarchy
    - Subtle shadows for depth
    - Professional staging that matches Fetch examples
    
    Args:
        loaded_images: List of (PIL.Image, filename) tuples
        image_type: Target image type (offer_tile, brand_hero, etc.)
        background_hex: Background color hex code
        product_callouts: Instructions for product arrangement
    
    Returns:
        PIL.Image: Professional composite matching Fetch quality standards
    """
    import random
    random.seed(42)  # Consistent layouts for same inputs
    
    # Get target dimensions
    target_width, target_height = IMAGE_TYPES[image_type]["size"]
    
    # Convert hex to RGB
    if background_hex.startswith('#'):
        hex_code = background_hex[1:]
    else:
        hex_code = background_hex
    
    try:
        bg_color = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        bg_color = (255, 255, 255)
    
    # Create base canvas
    composite = Image.new("RGBA", (target_width, target_height), bg_color + (255,))
    
    # Step 1: Remove backgrounds from all products
    print("📸 Removing backgrounds from products...")
    products_rgba = []
    for img, filename in loaded_images:
        # Remove background
        img_no_bg = remove_background(img)
        products_rgba.append((img_no_bg, filename))
    
    # Parse callouts
    callouts_lower = product_callouts.lower() if product_callouts else ""
    
    # Enhanced callout parsing: match products to instructions
    def parse_product_instructions(callouts, filename, index):
        """Parse callouts to extract instructions for a specific product"""
        instructions = {
            'size': 'medium',  # small, medium, large
            'position': 'auto',  # auto, center, left, right, top, bottom, front, back
            'priority': 1  # higher = more prominent
        }
        
        if not callouts:
            return instructions
        
        callouts = callouts.lower()
        filename_lower = filename.lower()
        
        # Try to find this product mentioned in callouts
        # Look for common product keywords or parts of filename
        filename_parts = filename_lower.replace('_', ' ').replace('-', ' ').replace('.jpg', '').replace('.png', '').replace('.jpeg', '').split()
        
        # Check if this product is mentioned
        is_mentioned = False
        mention_context = ""
        
        # Split callouts into sentences
        sentences = callouts.replace(',', '.').split('.')
        for sentence in sentences:
            # Check if any filename part is in this sentence
            if any(part in sentence and len(part) > 2 for part in filename_parts):
                is_mentioned = True
                mention_context = sentence
                break
        
        # Also check for ordinal mentions (first image, second product, etc.)
        ordinals = ['first', 'second', 'third', 'fourth', 'fifth']
        if index < len(ordinals) and ordinals[index] in callouts:
            is_mentioned = True
            # Find the sentence with this ordinal
            for sentence in sentences:
                if ordinals[index] in sentence:
                    mention_context = sentence
                    break
        
        # If not specifically mentioned, check for generic instructions
        if not is_mentioned:
            mention_context = callouts
        
        # Parse size instructions
        if any(word in mention_context for word in ['large', 'big', 'prominent', 'main', 'hero', 'feature']):
            instructions['size'] = 'large'
            instructions['priority'] = 3
        elif any(word in mention_context for word in ['small', 'tiny', 'minor']):
            instructions['size'] = 'small'
            instructions['priority'] = 0
        
        # Parse position instructions
        if any(word in mention_context for word in ['center', 'middle', 'centered']):
            instructions['position'] = 'center'
            instructions['priority'] = max(instructions['priority'], 2)
        elif 'left' in mention_context:
            instructions['position'] = 'left'
        elif 'right' in mention_context:
            instructions['position'] = 'right'
        elif 'top' in mention_context or 'above' in mention_context:
            instructions['position'] = 'top'
        elif 'bottom' in mention_context or 'below' in mention_context:
            instructions['position'] = 'bottom'
        elif any(word in mention_context for word in ['front', 'foreground', 'forward']):
            instructions['position'] = 'front'
            instructions['priority'] = max(instructions['priority'], 2)
        elif any(word in mention_context for word in ['back', 'background', 'behind']):
            instructions['position'] = 'back'
            instructions['priority'] = min(instructions['priority'], 0)
        
        return instructions
    
    # Parse instructions for all products
    product_instructions = []
    for idx, (img_rgba, filename) in enumerate(products_rgba):
        instructions = parse_product_instructions(product_callouts, filename, idx)
        product_instructions.append(instructions)
        print(f"  📋 Product {idx+1} ({filename}): size={instructions['size']}, position={instructions['position']}, priority={instructions['priority']}")
    
    # Step 2: Determine composition style based on count
    num_products = len(products_rgba)
    
    if num_products == 1:
        # SINGLE PRODUCT: Center-focused hero shot
        img_rgba, filename = products_rgba[0]
        instr = product_instructions[0]
        
        # Determine size based on parsed instructions
        if instr['size'] == 'large':
            scale = 0.75
        elif instr['size'] == 'small':
            scale = 0.45
        else:
            scale = 0.65  # Default medium size
        
        # Calculate dimensions maintaining aspect ratio
        img_aspect = img_rgba.width / img_rgba.height
        if img_aspect > 1:
            # Landscape
            new_width = int(target_width * scale)
            new_height = int(new_width / img_aspect)
        else:
            # Portrait or square
            new_height = int(target_height * scale)
            new_width = int(new_height * img_aspect)
        
        # Resize
        img_resized = img_rgba.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Add subtle shadow
        img_with_shadow = add_drop_shadow(img_resized, offset=(8, 8), blur_radius=12)
        
        # Position based on parsed instructions
        x = (target_width - img_with_shadow.width) // 2
        y = (target_height - img_with_shadow.height) // 2
        
        if instr['position'] == 'left':
            x = target_width // 6
        elif instr['position'] == 'right':
            x = target_width - img_with_shadow.width - target_width // 6
        elif instr['position'] == 'top':
            y = target_height // 6
        elif instr['position'] == 'bottom':
            y = target_height - img_with_shadow.height - target_height // 6
        # 'center' and 'auto' stay centered (default)
        
        composite.paste(img_with_shadow, (x, y), img_with_shadow)
    
    elif num_products == 2:
        # TWO PRODUCTS: Side-by-side showcase with priority-based sizing
        # Sort by priority to determine which gets more space
        products_with_priority = [(products_rgba[i], product_instructions[i], i) for i in range(2)]
        products_with_priority.sort(key=lambda x: x[1]['priority'], reverse=True)
        
        for idx, ((img_rgba, filename), instr, original_idx) in enumerate(products_with_priority):
            # Size based on priority and instructions
            if instr['priority'] >= 2:
                scale = 0.6
            elif instr['size'] == 'large':
                scale = 0.58
            elif instr['size'] == 'small':
                scale = 0.42
            else:
                scale = 0.5
            
            img_aspect = img_rgba.width / img_rgba.height
            max_height = int(target_height * 0.7)
            new_height = int(max_height * (scale / 0.5))
            new_width = int(new_height * img_aspect)
            
            # Adjust if too wide
            if new_width > target_width * 0.45:
                new_width = int(target_width * 0.45)
                new_height = int(new_width / img_aspect)
            
            img_resized = img_rgba.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_with_shadow = add_drop_shadow(img_resized, offset=(6, 6), blur_radius=10)
            
            # Position: respect instructions, default to side-by-side
            if instr['position'] == 'center':
                x = (target_width - img_with_shadow.width) // 2
            elif instr['position'] == 'left':
                x = target_width // 6
            elif instr['position'] == 'right':
                x = target_width - img_with_shadow.width - target_width // 6
            else:
                # Default side-by-side based on original index
                if original_idx == 0:
                    x = target_width // 4 - img_with_shadow.width // 2
                else:
                    x = 3 * target_width // 4 - img_with_shadow.width // 2
            
            # Vertical positioning
            if instr['position'] == 'top':
                y = target_height // 6
            elif instr['position'] == 'bottom':
                y = target_height - img_with_shadow.height - target_height // 6
            else:
                y = (target_height - img_with_shadow.height) // 2
            
            composite.paste(img_with_shadow, (x, y), img_with_shadow)
    
    elif num_products == 3:
        # THREE PRODUCTS: Featured center with flankers, respecting instructions
        # Identify which product should be centered based on priority
        products_with_data = [(products_rgba[i], product_instructions[i], i) for i in range(3)]
        products_with_data.sort(key=lambda x: x[1]['priority'], reverse=True)
        
        # Highest priority goes to center unless instructed otherwise
        center_idx = products_with_data[0][2]
        
        for i, (img_rgba, filename) in enumerate(products_rgba):
            instr = product_instructions[i]
            
            # Determine if this is the center product
            is_center = (i == center_idx) or instr['position'] == 'center'
            
            # Size based on role and instructions
            if is_center or instr['priority'] >= 2:
                if instr['size'] == 'large':
                    scale = 0.6
                else:
                    scale = 0.5
            elif instr['size'] == 'large':
                scale = 0.48
            elif instr['size'] == 'small':
                scale = 0.32
            else:
                scale = 0.4
            
            img_aspect = img_rgba.width / img_rgba.height
            max_height = int(target_height * scale)
            new_height = max_height
            new_width = int(new_height * img_aspect)
            
            img_resized = img_rgba.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_with_shadow = add_drop_shadow(img_resized, offset=(6, 6), blur_radius=10)
            
            # Position based on instructions
            if is_center and instr['position'] != 'left' and instr['position'] != 'right':
                x = (target_width - img_with_shadow.width) // 2
                y = (target_height - img_with_shadow.height) // 2
            elif instr['position'] == 'left' or (not is_center and i == 0):
                x = target_width // 6 - img_with_shadow.width // 2
                y = (target_height - img_with_shadow.height) // 2 + random.randint(-30, 30)
            elif instr['position'] == 'right' or (not is_center and i == 2):
                x = 5 * target_width // 6 - img_with_shadow.width // 2
                y = (target_height - img_with_shadow.height) // 2 + random.randint(-30, 30)
            else:
                # Fallback positioning
                if i == center_idx:
                    x = (target_width - img_with_shadow.width) // 2
                    y = (target_height - img_with_shadow.height) // 2
                elif i < center_idx:
                    x = target_width // 6 - img_with_shadow.width // 2
                    y = (target_height - img_with_shadow.height) // 2
                else:
                    x = 5 * target_width // 6 - img_with_shadow.width // 2
                    y = (target_height - img_with_shadow.height) // 2
            
            composite.paste(img_with_shadow, (x, y), img_with_shadow)
    
    elif num_products == 4:
        # FOUR PRODUCTS: Balanced quad layout respecting priority
        products_with_data = [(products_rgba[i], product_instructions[i], i) for i in range(4)]
        products_with_data.sort(key=lambda x: x[1]['priority'], reverse=True)
        
        # Default positions for 4 products
        positions = [
            (0.25, 0.3),  # Top-left
            (0.75, 0.3),  # Top-right
            (0.25, 0.7),  # Bottom-left
            (0.75, 0.7),  # Bottom-right
        ]
        
        for idx, (img_rgba, filename) in enumerate(products_rgba):
            instr = product_instructions[idx]
            
            # Size based on priority
            if instr['priority'] >= 2 or instr['size'] == 'large':
                scale = 0.45
            elif instr['size'] == 'small':
                scale = 0.3
            else:
                scale = random.uniform(0.35, 0.42)
            
            img_aspect = img_rgba.width / img_rgba.height
            max_size = int(min(target_width, target_height) * scale)
            
            if img_aspect > 1:
                new_width = max_size
                new_height = int(max_size / img_aspect)
            else:
                new_height = max_size
                new_width = int(max_size * img_aspect)
            
            img_resized = img_rgba.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_with_shadow = add_drop_shadow(img_resized, offset=(5, 5), blur_radius=8)
            
            # Position: check for specific instructions first
            if instr['position'] == 'center':
                x = (target_width - img_with_shadow.width) // 2
                y = (target_height - img_with_shadow.height) // 2
            else:
                # Use grid position with slight randomness
                base_x = int(positions[idx][0] * target_width)
                base_y = int(positions[idx][1] * target_height)
                
                x = base_x - img_with_shadow.width // 2 + random.randint(-20, 20)
                y = base_y - img_with_shadow.height // 2 + random.randint(-20, 20)
            
            composite.paste(img_with_shadow, (x, y), img_with_shadow)
    
    else:
        # 5+ PRODUCTS: Dynamic collage with priority-based featuring
        # Identify featured products based on parsed priority
        products_with_data = [(products_rgba[i], product_instructions[i], i) for i in range(len(products_rgba))]
        products_with_data.sort(key=lambda x: x[1]['priority'], reverse=True)
        
        # Top 1-2 priority products are featured
        featured_indices = [products_with_data[i][2] for i in range(min(2, len(products_with_data))) 
                           if products_with_data[i][1]['priority'] >= 2]
        
        # If no high priority, feature first 1-2
        if not featured_indices:
            featured_indices = [0] if num_products == 5 else [0, 1]
        
        placed_rects = []
        
        # Place featured products first (larger, centered)
        for idx in featured_indices[:2]:
            img_rgba, filename = products_rgba[idx]
            instr = product_instructions[idx]
            
            # Size based on instructions
            if instr['size'] == 'large':
                scale = random.uniform(0.5, 0.6)
            else:
                scale = random.uniform(0.45, 0.55)
            
            img_aspect = img_rgba.width / img_rgba.height
            max_size = int(min(target_width, target_height) * scale)
            
            if img_aspect > 1:
                new_width = max_size
                new_height = int(max_size / img_aspect)
            else:
                new_height = max_size
                new_width = int(max_size * img_aspect)
            
            img_resized = img_rgba.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_with_shadow = add_drop_shadow(img_resized, offset=(7, 7), blur_radius=10)
            
            # Position based on instructions
            if instr['position'] == 'center' or instr['position'] == 'auto':
                x = (target_width - img_with_shadow.width) // 2 + random.randint(-40, 40)
                y = (target_height - img_with_shadow.height) // 2 + random.randint(-40, 40)
            elif instr['position'] == 'left':
                x = target_width // 4 - img_with_shadow.width // 2
                y = (target_height - img_with_shadow.height) // 2 + random.randint(-40, 40)
            elif instr['position'] == 'right':
                x = 3 * target_width // 4 - img_with_shadow.width // 2
                y = (target_height - img_with_shadow.height) // 2 + random.randint(-40, 40)
            else:
                x = (target_width - img_with_shadow.width) // 2 + random.randint(-40, 40)
                y = (target_height - img_with_shadow.height) // 2 + random.randint(-40, 40)
            
            composite.paste(img_with_shadow, (x, y), img_with_shadow)
            placed_rects.append((x, y, x + img_with_shadow.width, y + img_with_shadow.height))
        
        # Place remaining products around featured ones
        for idx, (img_rgba, filename) in enumerate(products_rgba):
            if idx in featured_indices[:2]:
                continue
            
            instr = product_instructions[idx]
            
            # Size based on instructions and role
            if instr['size'] == 'large':
                scale = random.uniform(0.32, 0.4)
            elif instr['size'] == 'small':
                scale = random.uniform(0.18, 0.26)
            else:
                scale = random.uniform(0.25, 0.35)
            
            img_aspect = img_rgba.width / img_rgba.height
            max_size = int(min(target_width, target_height) * scale)
            
            if img_aspect > 1:
                new_width = max_size
                new_height = int(max_size / img_aspect)
            else:
                new_height = max_size
                new_width = int(max_size * img_aspect)
            
            img_resized = img_rgba.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_with_shadow = add_drop_shadow(img_resized, offset=(4, 4), blur_radius=6)
            
            # Find non-overlapping position
            attempts = 0
            while attempts < 100:
                # Random position in outer areas
                if random.random() < 0.5:
                    # Left or right edge
                    x = random.randint(0, target_width // 4) if random.random() < 0.5 else \
                        random.randint(3 * target_width // 4, target_width - img_with_shadow.width)
                    y = random.randint(0, target_height - img_with_shadow.height)
                else:
                    # Top or bottom edge
                    x = random.randint(0, target_width - img_with_shadow.width)
                    y = random.randint(0, target_height // 4) if random.random() < 0.5 else \
                        random.randint(3 * target_height // 4, target_height - img_with_shadow.height)
                
                new_rect = (x, y, x + img_with_shadow.width, y + img_with_shadow.height)
                
                # Check overlap
                overlap = False
                for rect in placed_rects:
                    # Allow small overlap for natural look
                    overlap_threshold = 30
                    if not (new_rect[2] < rect[0] - overlap_threshold or 
                           new_rect[0] > rect[2] + overlap_threshold or 
                           new_rect[3] < rect[1] - overlap_threshold or 
                           new_rect[1] > rect[3] + overlap_threshold):
                        overlap = True
                        break
                
                if not overlap:
                    composite.paste(img_with_shadow, (x, y), img_with_shadow)
                    placed_rects.append(new_rect)
                    break
                
                attempts += 1
    
    # Convert back to RGB for final output
    final_composite = Image.new("RGB", (target_width, target_height), bg_color)
    final_composite.paste(composite, (0, 0), composite)
    
    # Apply Fetch specifications
    final_composite = enforce_specs(final_composite, image_type)
    
    return final_composite


def build_prompt(preset_key, brand, products, background_hex, lifestyle_keywords=""):
    if preset_key == "offer_tile":
        return (
          f"Create a Fetch style offer tile featuring {', '.join(products)} on a solid {background_hex} background. "
          "1120x1120 JPG. Use 3 to 5 products, balanced arrangement, labels legible, bright consistent lighting, "
          "realistic photo look, no lifestyle props, no text, no extra logos, no gradients, no watermark."
        )
    if preset_key == "offer_detail":
        return (
          f"Generate a lifestyle product photo for {brand} {', '.join(products)}. "
          "1120x1120 JPEG. Real environment like tabletop or ingredients, authentic composition, warm natural light, "
          "labels readable, realistic photography only, no added graphics or watermark."
        )
    if preset_key == "brand_hero":
        return (
          f"Create a brand lifestyle image for {brand}. 1200x857 JPG. "
          "Optional product inclusion, warm consistent lighting, realistic photography, no overlays."
        )
    if preset_key == "brand_logo":
        return (
          f"Centered brand logo for {brand} on solid high contrast {background_hex}. "
          "400x400 PNG. Circle safe, no transparency, no borders, no extra text."
        )
    raise ValueError("Unknown preset")

# stub generator for now: returns blank white canvas at the requested size
def generate_image_stub(preset_key):
    w, h = IMAGE_TYPES[preset_key]["size"]
    return Image.new("RGB", (w, h), "#FFFFFF")

def enforce_specs(img, preset_key):
    w, h = IMAGE_TYPES[preset_key]["size"]
    if img.size != (w, h):
        img = img.resize((w, h))
    return img.convert("RGB")

def _detect_text(img):
    try:
        txt = pytesseract.image_to_string(img)
        return bool(txt and txt.strip())
    except Exception:
        return False

def validate_image(img, preset_key, background_hex):
    spec = IMAGE_TYPES[preset_key]
    report = {"pass": True, "checks": []}

    # size
    ok_size = img.size == spec["size"]
    report["checks"].append({"name":"size", "ok": ok_size, "expected": spec["size"], "actual": img.size})
    report["pass"] &= ok_size

    # approved palette
    ok_palette = background_hex.upper() in APPROVED_HEX
    report["checks"].append({"name":"palette_hex", "ok": ok_palette, "actual": background_hex.upper()})
    report["pass"] &= ok_palette

    # no text overlays
    has_text = _detect_text(img)
    report["checks"].append({"name":"no_text", "ok": not has_text})
    report["pass"] &= (not has_text)

    # simple gradient guardrail: compare far left and far right columns
    left = ImageStat.Stat(img.crop((0,0,1,img.height))).mean
    right = ImageStat.Stat(img.crop((img.width-1,0,img.width,img.height))).mean
    ok_no_gradient_edges = abs(sum(left) - sum(right)) < 10
    report["checks"].append({"name":"no_strong_gradient_edges", "ok": ok_no_gradient_edges})
    report["pass"] &= ok_no_gradient_edges

    # output filename and mime
    ext = "png" if spec["format"] == "png" else "jpg"
    report["filename"] = f"{preset_key}_{background_hex[1:]}.{ext}"
    report["mime"] = "image/png" if ext == "png" else "image/jpeg"
    return report

def export_file(img, mime):
    buf = io.BytesIO()
    if mime == "image/png":
        img.save(buf, format="PNG")
    else:
        img.save(buf, format="JPEG", quality=92, optimize=True)
    return buf.getvalue()
