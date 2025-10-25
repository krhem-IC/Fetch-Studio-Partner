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
    
    # Step 2: Determine composition style based on count
    num_products = len(products_rgba)
    
    if num_products == 1:
        # SINGLE PRODUCT: Center-focused hero shot
        img_rgba, filename = products_rgba[0]
        
        # Determine size based on callouts
        if "large" in callouts_lower or "big" in callouts_lower:
            scale = 0.75
        elif "small" in callouts_lower:
            scale = 0.5
        else:
            scale = 0.65  # Default comfortable size
        
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
        
        # Center position (or adjust based on callouts)
        x = (target_width - img_with_shadow.width) // 2
        y = (target_height - img_with_shadow.height) // 2
        
        if "left" in callouts_lower:
            x = target_width // 6
        elif "right" in callouts_lower:
            x = target_width - img_with_shadow.width - target_width // 6
        elif "top" in callouts_lower:
            y = target_height // 6
        elif "bottom" in callouts_lower:
            y = target_height - img_with_shadow.height - target_height // 6
        
        composite.paste(img_with_shadow, (x, y), img_with_shadow)
    
    elif num_products == 2:
        # TWO PRODUCTS: Side-by-side showcase
        for i, (img_rgba, filename) in enumerate(products_rgba):
            # Slightly different sizes for visual interest
            scale = 0.55 if i == 0 else 0.5
            
            img_aspect = img_rgba.width / img_rgba.height
            max_height = int(target_height * 0.7)
            new_height = max_height
            new_width = int(new_height * img_aspect)
            
            # Adjust if too wide
            if new_width > target_width * 0.4:
                new_width = int(target_width * 0.4)
                new_height = int(new_width / img_aspect)
            
            img_resized = img_rgba.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_with_shadow = add_drop_shadow(img_resized, offset=(6, 6), blur_radius=10)
            
            # Position: left and right thirds
            if i == 0:
                x = target_width // 4 - img_with_shadow.width // 2
            else:
                x = 3 * target_width // 4 - img_with_shadow.width // 2
            
            y = (target_height - img_with_shadow.height) // 2
            
            composite.paste(img_with_shadow, (x, y), img_with_shadow)
    
    elif num_products == 3:
        # THREE PRODUCTS: Featured center with flankers
        for i, (img_rgba, filename) in enumerate(products_rgba):
            # Center product larger
            if i == 1 or "center" in filename.lower() or "main" in filename.lower():
                scale = 0.5
                is_center = True
            else:
                scale = 0.4
                is_center = False
            
            img_aspect = img_rgba.width / img_rgba.height
            max_height = int(target_height * scale)
            new_height = max_height
            new_width = int(new_height * img_aspect)
            
            img_resized = img_rgba.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_with_shadow = add_drop_shadow(img_resized, offset=(6, 6), blur_radius=10)
            
            # Position: center with flankers
            if is_center:
                x = (target_width - img_with_shadow.width) // 2
                y = (target_height - img_with_shadow.height) // 2
            elif i == 0:
                x = target_width // 6 - img_with_shadow.width // 2
                y = (target_height - img_with_shadow.height) // 2 + random.randint(-30, 30)
            else:
                x = 5 * target_width // 6 - img_with_shadow.width // 2
                y = (target_height - img_with_shadow.height) // 2 + random.randint(-30, 30)
            
            composite.paste(img_with_shadow, (x, y), img_with_shadow)
    
    elif num_products == 4:
        # FOUR PRODUCTS: Balanced quad layout
        positions = [
            (0.25, 0.3),  # Top-left
            (0.75, 0.3),  # Top-right
            (0.25, 0.7),  # Bottom-left
            (0.75, 0.7),  # Bottom-right
        ]
        
        for i, (img_rgba, filename) in enumerate(products_rgba):
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
            
            # Position based on grid with slight randomness
            base_x = int(positions[i][0] * target_width)
            base_y = int(positions[i][1] * target_height)
            
            x = base_x - img_with_shadow.width // 2 + random.randint(-20, 20)
            y = base_y - img_with_shadow.height // 2 + random.randint(-20, 20)
            
            composite.paste(img_with_shadow, (x, y), img_with_shadow)
    
    else:
        # 5+ PRODUCTS: Dynamic collage with featured items
        # Identify featured products from callouts
        featured_indices = []
        if product_callouts:
            for idx, (_, filename) in enumerate(products_rgba):
                filename_lower = filename.lower()
                if any(word in filename_lower for word in callouts_lower.split()) and \
                   any(keyword in callouts_lower for keyword in ["center", "main", "feature", "hero"]):
                    featured_indices.append(idx)
        
        # If no featured found, use first 1-2
        if not featured_indices:
            featured_indices = [0] if num_products == 5 else [0, 1]
        
        placed_rects = []
        
        # Place featured products first (larger, centered)
        for idx in featured_indices[:2]:
            img_rgba, filename = products_rgba[idx]
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
            
            # Center with slight offset
            x = (target_width - img_with_shadow.width) // 2 + random.randint(-40, 40)
            y = (target_height - img_with_shadow.height) // 2 + random.randint(-40, 40)
            
            composite.paste(img_with_shadow, (x, y), img_with_shadow)
            placed_rects.append((x, y, x + img_with_shadow.width, y + img_with_shadow.height))
        
        # Place remaining products around featured ones
        for idx, (img_rgba, filename) in enumerate(products_rgba):
            if idx in featured_indices[:2]:
                continue
            
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


def generate_image_stub(preset_key):
    
    else:
        # Many images: create a collage effect with callout consideration
        import random
        random.seed(42)  # Consistent layout
        
        # Parse for featured items
        featured_items = []
        regular_items = []
        
        if product_callouts:
            callouts_words = callouts_lower.split()
            for img, filename in loaded_images:
                filename_lower = filename.lower()
                is_featured = any(word in filename_lower and 
                                ("center" in callouts_lower or "main" in callouts_lower or "feature" in callouts_lower)
                                for word in callouts_words)
                if is_featured:
                    featured_items.append((img, filename))
                else:
                    regular_items.append((img, filename))
        else:
            regular_items = loaded_images
        
        # Place featured items first with STRICT aspect ratio preservation
        placed_rects = []
        
        for img, filename in featured_items[:2]:  # Max 2 featured items
            base_size = min(target_width, target_height) // 3  # Base size for featured
            
            # CRITICAL: Calculate exact dimensions preserving aspect ratio
            img_aspect = img.width / img.height
            if img_aspect > 1:
                # Landscape image
                new_width = base_size
                new_height = int(base_size / img_aspect)
            else:
                # Portrait or square image
                new_height = base_size
                new_width = int(base_size * img_aspect)
            
            # Ensure dimensions are valid and within canvas
            new_width = max(1, min(new_width, target_width))
            new_height = max(1, min(new_height, target_height))
            
            # Resize preserving exact proportions
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Place in center area
            x = (target_width - new_width) // 2
            y = (target_height - new_height) // 2
            
            composite.paste(resized_img, (x, y))
            placed_rects.append((x, y, x + new_width, y + new_height))
        
        # Place regular items around featured items with preserved proportions
        max_img_size = min(target_width, target_height) // 5  # Smaller regular items
        
        for img, filename in regular_items[:6]:  # Limit regular items
            attempts = 0
            while attempts < 50:
                base_size = random.randint(max_img_size // 2, max_img_size)
                
                # CRITICAL: Calculate exact dimensions preserving aspect ratio
                img_aspect = img.width / img.height
                if img_aspect > 1:
                    # Landscape image
                    new_width = base_size
                    new_height = int(base_size / img_aspect)
                else:
                    # Portrait or square image
                    new_height = base_size
                    new_width = int(base_size * img_aspect)
                
                # Ensure dimensions are valid and within canvas
                new_width = max(1, min(new_width, target_width))
                new_height = max(1, min(new_height, target_height))
                
                # Skip if dimensions are too small to be meaningful
                if new_width < 10 or new_height < 10:
                    attempts += 1
                    continue
                
                # Resize preserving exact proportions
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                x = random.randint(0, max(0, target_width - new_width))
                y = random.randint(0, max(0, target_height - new_height))
                
                new_rect = (x, y, x + new_width, y + new_height)
                
                # Check for overlap
                overlap = False
                for rect in placed_rects:
                    if not (new_rect[2] < rect[0] or new_rect[0] > rect[2] or 
                           new_rect[3] < rect[1] or new_rect[1] > rect[3]):
                        overlap = True
                        break
                
                if not overlap:
                    composite.paste(resized_img, (x, y))
                    placed_rects.append(new_rect)
                    break
                
                attempts += 1
    
    # FINAL STEP: Ensure composite meets exact Fetch specifications
    final_composite = enforce_specs(composite, image_type)
    
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
