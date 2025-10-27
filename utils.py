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
    Remove backgrounds using AI with optimized settings for speed and quality.
    Handles both white backgrounds and complex backgrounds effectively.
    Preserves thin details like straws, ribbons, and small accessories.
    
    Args:
        img: PIL Image in RGB mode
    
    Returns:
        PIL.Image: Image with transparent background (RGBA mode)
    """
    try:
        # Resize large images to speed up processing
        max_dimension = 2000
        original_size = img.size
        
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"  ↓ Resized image to {new_size} for faster processing")
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Check if it's primarily a white background (simple case)
        # Sample corners to detect white background
        corners = [
            img.getpixel((0, 0)),
            img.getpixel((img.width-1, 0)),
            img.getpixel((0, img.height-1)),
            img.getpixel((img.width-1, img.height-1))
        ]
        
        # If all corners are very light (likely white background)
        is_white_bg = all(all(c > 240 for c in pixel) for pixel in corners)
        
        if is_white_bg:
            # Use simple white background removal to preserve thin details
            print(f"  📋 Detected white background - using detail-preserving removal")
            img_rgba = img.convert('RGBA')
            data = img_rgba.getdata()
            
            new_data = []
            for item in data:
                # Make ONLY pure white/near-white pixels transparent
                # Very strict threshold to preserve white text/logos on products
                if item[0] > 248 and item[1] > 248 and item[2] > 248:
                    # Very light - likely background
                    new_data.append((255, 255, 255, 0))
                else:
                    # Keep original pixel (including off-white text/labels)
                    new_data.append(item)
            
            img_rgba.putdata(new_data)
            img_no_bg = img_rgba
        else:
            # Use AI removal for complex backgrounds
            print(f"  🤖 Using AI background removal")
            
            # Use AI but be more conservative to preserve text/logos
            img_no_bg = rembg_remove(
                img,
                post_process_mask=True,  # Clean up mask edges
                bgcolor=(0, 0, 0, 0)  # Transparent background
            )
        
        # Check if background was actually removed
        if img_no_bg.mode == 'RGBA':
            alpha = img_no_bg.split()[3]
            bbox = alpha.getbbox()
            
            if bbox is None:
                # All transparent - something went wrong, use original
                print(f"  ⚠️  Background removal resulted in empty image, using original")
                if original_size != img.size:
                    img = img.resize(original_size, Image.Resampling.LANCZOS)
                return img.convert('RGBA')
            
            # Check if we lost too much content (might have removed thin details)
            alpha_pixels = list(alpha.getdata())
            non_transparent = sum(1 for p in alpha_pixels if p > 10)
            total_pixels = len(alpha_pixels)
            content_ratio = non_transparent / total_pixels
            
            if content_ratio < 0.05:
                # Less than 5% visible - probably lost important details
                print(f"  ⚠️  Too much removed ({content_ratio:.1%} visible), retrying with white removal")
                # Fallback to simple white removal
                img_rgba = img.convert('RGBA')
                data = img_rgba.getdata()
                
                new_data = []
                for item in data:
                    if item[0] > 235 and item[1] > 235 and item[2] > 235:
                        new_data.append((255, 255, 255, 0))
                    else:
                        new_data.append(item)
                
                img_rgba.putdata(new_data)
                img_no_bg = img_rgba
        
        # Resize back to original size if we downsized
        if original_size != img.size and img_no_bg.size != original_size:
            img_no_bg = img_no_bg.resize(original_size, Image.Resampling.LANCZOS)
        
        return img_no_bg
        
    except Exception as e:
        print(f"  ⚠️  Background removal failed: {e}. Using original image.")
        # Return original image with alpha channel
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
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


def create_composite_image(loaded_images, image_type, background_hex, product_callouts="", logo_option=None):
    """
    Create a Fetch-style composite image with natural product arrangement.
    
    FLEXIBLE CALLOUT SYSTEM:
    - NO CALLOUTS (blank): Uses professional default layouts optimized for each product count
    - PARTIAL CALLOUTS: Respects specific instructions, uses smart defaults for unmentioned products
    - DETAILED CALLOUTS: Full control over every product's size, position, and priority
    
    Callout Examples:
    - Blank: Clean, balanced, professional layouts
    - Simple: "make it large" or "center it"
    - Specific: "put the 2 liter bottle in the middle, keep others small"
    - Detailed: "first product large and centered, second on left, third small in background"
    
    Products are automatically background-removed and composed with:
    - Natural, varied positioning (not rigid grids)
    - Size variation for visual hierarchy
    - Subtle shadows for depth
    - Professional staging that matches Fetch examples
    
    Args:
        loaded_images: List of (PIL.Image, filename) tuples
        image_type: Target image type (offer_tile, brand_hero, etc.)
        background_hex: Background color hex code
        product_callouts: Optional instructions for product arrangement (can be blank)
        logo_option: For brand_logo type - "resize" or "add_white" (optional)
    
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
    
    # Special handling for brand logos - check BEFORE background removal
    if image_type == "brand_logo":
        print("🎨 Processing brand logo...")
        
        # Get ORIGINAL image
        original_logo_img, logo_filename = loaded_images[0]
        
        # Check logo option
        if logo_option and "just resize" in logo_option.lower():
            # Logo already has colored background - JUST RESIZE, no processing
            print("  📐 Resizing logo with existing background (no alterations)")
            
            # Resize to exact Fetch specs: 400x400
            logo_resized = original_logo_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Convert to RGB for output
            if logo_resized.mode == 'RGBA':
                final_composite = Image.new("RGB", (target_width, target_height), (255, 255, 255))
                final_composite.paste(logo_resized, (0, 0), logo_resized)
            else:
                final_composite = logo_resized.convert('RGB')
            
            return final_composite
    
    # Step 1: Remove backgrounds from all products (skip for "just resize" logos)
    print("📸 Removing backgrounds from products...")
    products_rgba = []
    for i, (img, filename) in enumerate(loaded_images):
        print(f"  Processing {i+1}/{len(loaded_images)}: {filename}")
        try:
            # Remove background - use LESS aggressive method for logos
            if image_type == "brand_logo":
                # For logos, be very conservative - only remove pure white
                print(f"  🎨 Using conservative background removal for logo")
                img_rgba = img.convert('RGBA')
                data = img_rgba.getdata()
                
                new_data = []
                for item in data:
                    # Only remove PURE white (>250 on all channels)
                    if item[0] > 250 and item[1] > 250 and item[2] > 250:
                        new_data.append((255, 255, 255, 0))
                    else:
                        # Keep everything else including off-white text
                        new_data.append(item)
                
                img_rgba.putdata(new_data)
                img_no_bg = img_rgba
            else:
                # Standard background removal for products
                img_no_bg = remove_background(img)
            
            products_rgba.append((img_no_bg, filename))
            print(f"  ✓ Completed {filename}")
        except Exception as e:
            print(f"  ✗ Failed to process {filename}: {str(e)}")
            # Use original image with white background removed as fallback
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            products_rgba.append((img, filename))
    
    if not products_rgba:
        raise ValueError("No images could be processed")
    
    print(f"\n✅ Successfully processed {len(products_rgba)} products")
    
    # Continue logo handling for white/colored background options
    if image_type == "brand_logo":
        if logo_option and "white background" in logo_option.lower():
            # Add white background to logo (transparent logo + white = colored background)
            print("  ⬜ Adding white background to transparent logo")
            
            # Use background-removed version from products_rgba
            logo_img, _ = products_rgba[0]
            
            # Create white canvas at exact size
            white_canvas = Image.new('RGB', (target_width, target_height), (255, 255, 255))
            
            # Calculate logo size (leave some padding)
            padding = 50
            max_width = target_width - (padding * 2)
            max_height = target_height - (padding * 2)
            
            logo_aspect = logo_img.width / logo_img.height
            if logo_aspect > (max_width / max_height):
                new_width = max_width
                new_height = int(max_width / logo_aspect)
            else:
                new_height = max_height
                new_width = int(max_height * logo_aspect)
            
            logo_resized = logo_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Center logo on white canvas
            x = (target_width - new_width) // 2
            y = (target_height - new_height) // 2
            
            white_canvas.paste(logo_resized, (x, y), logo_resized if logo_resized.mode == 'RGBA' else None)
            
            return white_canvas
            
        elif logo_option and "colored background" in logo_option.lower():
            # Third option: colored background (use selected color)
            print(f"  🎨 Adding {background_hex} background to logo")
            
            # Use background-removed version
            logo_img, _ = products_rgba[0]
            
            # Create colored canvas
            colored_canvas = Image.new('RGB', (target_width, target_height), bg_color)
            
            # Calculate logo size (leave some padding)
            padding = 50
            max_width = target_width - (padding * 2)
            max_height = target_height - (padding * 2)
            
            logo_aspect = logo_img.width / logo_img.height
            if logo_aspect > (max_width / max_height):
                new_width = max_width
                new_height = int(max_width / logo_aspect)
            else:
                new_height = max_height
                new_width = int(max_height * logo_aspect)
            
            logo_resized = logo_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Center logo on colored canvas
            x = (target_width - new_width) // 2
            y = (target_height - new_height) // 2
            
            colored_canvas.paste(logo_resized, (x, y), logo_resized if logo_resized.mode == 'RGBA' else None)
            
            return colored_canvas
    
    # Parse callouts (works with blank, partial, or detailed instructions)
    callouts_lower = product_callouts.lower() if product_callouts else ""
    
    if not product_callouts or not product_callouts.strip():
        print("ℹ️  No callouts provided - using professional default layouts")
    else:
        print(f"📝 Parsing callouts: '{product_callouts}'")
    
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
        
        # Common product descriptors with specificity levels
        product_descriptors = {
            '2 liter': ['2 liter', '2l', '2-liter', 'two liter'],
            'bottle': ['bottle', 'beverage', 'drink'],
            'can': ['can', 'soda', 'pop'],
            'box': ['box', 'cereal', 'package', 'carton'],
            'multipack': ['multipack', 'multi-pack', 'pack', 'case', 'variety'],
            'bag': ['bag', 'pouch', 'snack'],
            'container': ['container', 'tub', 'jar'],
            'gift set': ['gift set', 'set', 'kit'],
        }
        
        # Try to find this product mentioned in callouts
        # Look for common product keywords or parts of filename
        filename_parts = filename_lower.replace('_', ' ').replace('-', ' ').replace('.jpg', '').replace('.png', '').replace('.jpeg', '').split()
        
        # Check if this product is mentioned - be MORE SPECIFIC
        is_mentioned = False
        mention_context = ""
        match_score = 0  # Higher score = more specific match
        
        # Split callouts into sentences
        sentences = callouts.replace(',', '.').replace(';', '.').split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if any filename part is in this sentence (strong match)
            filename_match = any(part in sentence and len(part) > 2 for part in filename_parts)
            if filename_match:
                is_mentioned = True
                mention_context = sentence
                match_score = 10  # Strong match
                break
            
            # Check for product descriptors - prioritize specific ones
            temp_match_score = 0
            temp_mentioned = False
            temp_context = ""
            
            for descriptor_type, keywords in product_descriptors.items():
                for keyword in keywords:
                    if keyword in sentence:
                        # Calculate specificity score
                        specificity = len(keyword.split())  # More words = more specific
                        if specificity > temp_match_score:
                            temp_match_score = specificity
                            temp_mentioned = True
                            temp_context = sentence
            
            # Only accept this match if it's better than current match
            if temp_mentioned and temp_match_score > match_score:
                is_mentioned = temp_mentioned
                mention_context = temp_context
                match_score = temp_match_score
        
        # Also check for ordinal mentions (first image, second product, etc.)
        ordinals = ['first', 'second', 'third', 'fourth', 'fifth']
        if index < len(ordinals) and ordinals[index] in callouts:
            is_mentioned = True
            match_score = 8  # Medium-strong match
            # Find the sentence with this ordinal
            for sentence in sentences:
                if ordinals[index] in sentence:
                    mention_context = sentence
                    break
        
        # If match score is too low (generic match), don't apply instructions
        # This prevents "bottle" from matching ALL bottles
        if is_mentioned and match_score < 2:
            return instructions  # Return default
        
        # If not specifically mentioned, leave as default (don't apply global instructions)
        if not is_mentioned:
            return instructions
        
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
    
    # CRITICAL: Deduplicate instructions - if multiple products have the same non-default position,
    # only keep it for the FIRST one (prevents all bottles going to center)
    position_used = {}
    for idx, instr in enumerate(product_instructions):
        pos = instr['position']
        if pos != 'auto':  # Only deduplicate specific positions
            if pos in position_used:
                # Already assigned to another product, reset this one to auto
                print(f"  ⚠️  Product {idx+1}: Resetting duplicate position '{pos}' to 'auto'")
                instr['position'] = 'auto'
            else:
                position_used[pos] = idx
    
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
        # TWO PRODUCTS: Tight side-by-side like Mr. Pibb reference
        # Products almost touching, balanced, no cutoff
        
        print("📐 Layout: 2-product tight side-by-side")
        
        # Define safe zone to prevent cutoff (85% of canvas)
        safe_width = int(target_width * 0.85)
        safe_height = int(target_height * 0.85)
        
        # Process both products with conservative sizing to prevent cutoff
        products_with_priority = [(products_rgba[i], product_instructions[i], i) for i in range(2)]
        products_with_priority.sort(key=lambda x: x[1]['priority'], reverse=True)
        
        processed_products = []
        for idx, ((img_rgba, filename), instr, original_idx) in enumerate(products_with_priority):
            # REDUCED sizing to prevent cutoff - much smaller than before
            if instr['priority'] >= 2 or instr['size'] == 'large':
                scale = 0.58  # Featured (was 0.75)
            elif instr['size'] == 'small':
                scale = 0.45  # Small (was 0.58)
            else:
                scale = 0.52  # Default medium (was 0.70)
            
            img_aspect = img_rgba.width / img_rgba.height
            
            # Calculate size preserving aspect ratio
            max_height = int(safe_height * scale)  # Use safe_height
            new_height = max_height
            new_width = int(new_height * img_aspect)
            
            # Ensure fits within safe zone width
            max_width = int(safe_width * 0.48)
            if new_width > max_width:
                new_width = max_width
                new_height = int(new_width / img_aspect)
            
            img_resized = img_rgba.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Standardized shadows for consistency
            img_with_shadow = add_drop_shadow(img_resized, offset=(8, 8), blur_radius=12)
            
            processed_products.append({
                'image': img_with_shadow,
                'instr': instr,
                'original_idx': original_idx
            })
        
        # MINIMAL gap - almost touching like Mott's reference
        gap = int(target_width * 0.005)  # 0.5% gap - essentially touching
        
        img1 = processed_products[0]['image']
        img2 = processed_products[1]['image']
        
        # Total width of both products plus minimal gap
        total_width = img1.width + gap + img2.width
        
        # Center the pair as a tight unit
        start_x = (target_width - total_width) // 2
        
        # Position products
        for idx, prod in enumerate(processed_products):
            img_with_shadow = prod['image']
            instr = prod['instr']
            original_idx = prod['original_idx']
            
            # Horizontal positioning - tight side by side
            if instr['position'] == 'center':
                x = (target_width - img_with_shadow.width) // 2
            elif instr['position'] == 'left':
                x = (target_width - safe_width) // 2 + safe_width // 6
            elif instr['position'] == 'right':
                x = target_width - (target_width - safe_width) // 2 - safe_width // 6 - img_with_shadow.width
            else:
                # Default: tight side-by-side within safe zone
                if original_idx == 0:
                    x = start_x
                else:
                    x = start_x + img1.width + gap
            
            # Vertical centering within safe zone
            if instr['position'] == 'top':
                y = (target_height - safe_height) // 2 + safe_height // 8
            elif instr['position'] == 'bottom':
                y = target_height - (target_height - safe_height) // 2 - safe_height // 8 - img_with_shadow.height
            else:
                # Center vertically for clean, professional look
                y = (target_height - img_with_shadow.height) // 2
            
            composite.paste(img_with_shadow, (x, y), img_with_shadow)
            print(f"  Placed product {original_idx + 1} at ({x}, {y}), size {img_with_shadow.size}")
    
    elif num_products == 3:
        # THREE PRODUCTS: Tight cluster arrangement
        # Professional arrangement like Gold Peak reference
        
        print("📐 Layout: 3-product tight cluster")
        
        # Safe zone: 85% of canvas to prevent cutoff
        safe_width = int(target_width * 0.85)
        safe_height = int(target_height * 0.85)
        
        # Identify hero product (highest priority or middle)
        products_with_data = [(products_rgba[i], product_instructions[i], i) for i in range(3)]
        products_with_data.sort(key=lambda x: x[1]['priority'], reverse=True)
        
        hero_idx = products_with_data[0][2] if products_with_data[0][1]['priority'] > 1 else 1
        
        processed_products = []
        for i, (img_rgba, filename) in enumerate(products_rgba):
            instr = product_instructions[i]
            is_hero = (i == hero_idx) or instr['position'] == 'center'
            
            # Reduced sizes for tight cluster - hero larger, flankers smaller
            if is_hero or instr['priority'] >= 2:
                scale = 0.54  # Hero - moderate size
            elif instr['size'] == 'large':
                scale = 0.48
            elif instr['size'] == 'small':
                scale = 0.35
            else:
                scale = 0.42  # Flanker - smaller
            
            img_aspect = img_rgba.width / img_rgba.height
            max_height = int(safe_height * scale)
            new_height = max_height
            new_width = int(new_height * img_aspect)
            
            # Ensure not too wide within safe zone
            max_width = int(safe_width * 0.38)
            if new_width > max_width:
                new_width = max_width
                new_height = int(new_width / img_aspect)
            
            img_resized = img_rgba.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_with_shadow = add_drop_shadow(img_resized, offset=(8, 8), blur_radius=12)
            
            processed_products.append({
                'image': img_with_shadow,
                'instr': instr,
                'is_hero': is_hero,
                'index': i
            })
        
        # MINIMAL gap - products almost touching like reference
        gap = int(target_width * 0.005)  # 0.5% gap
        
        for prod in processed_products:
            img_with_shadow = prod['image']
            instr = prod['instr']
            is_hero = prod['is_hero']
            i = prod['index']
            
            # Specific position override with safe zone offsets
            if instr['position'] == 'left':
                x = (target_width - safe_width) // 2 + safe_width // 6
                y = (target_height - img_with_shadow.height) // 2
            elif instr['position'] == 'right':
                x = target_width - (target_width - safe_width) // 2 - safe_width // 6 - img_with_shadow.width
                y = (target_height - img_with_shadow.height) // 2
            elif is_hero and instr['position'] not in ['left', 'right']:
                # Hero: centered
                x = (target_width - img_with_shadow.width) // 2
                y = (target_height - img_with_shadow.height) // 2
            else:
                # Flankers: tight to hero on left/right within safe zone
                hero_prod = next(p for p in processed_products if p['is_hero'])
                hero_width = hero_prod['image'].width
                center_x = target_width // 2
                
                if i < hero_idx:
                    # Left flanker - almost touching hero
                    x = center_x - hero_width // 2 - gap - img_with_shadow.width
                else:
                    # Right flanker - almost touching hero
                    x = center_x + hero_width // 2 + gap
                
                # Center vertically for clean look
                y = (target_height - img_with_shadow.height) // 2
            
            composite.paste(img_with_shadow, (x, y), img_with_shadow)
            print(f"  Placed product {i + 1} ({'HERO' if is_hero else 'flanker'}) at ({x}, {y}), size {img_with_shadow.size}")
    
    elif num_products == 4:
        # FOUR PRODUCTS: Tight 2x2 grid
        # Professional, balanced arrangement within safe zone
        
        print("📐 Layout: 4-product tight grid")
        
        # Safe zone: 85% of canvas to prevent cutoff
        safe_width = int(target_width * 0.85)
        safe_height = int(target_height * 0.85)
        
        # Process all products with reduced sizes
        processed_products = []
        for i, (img_rgba, filename) in enumerate(products_rgba):
            instr = product_instructions[i]
            
            # Reduced sizes for tight grid
            if instr['priority'] >= 2 or instr['size'] == 'large':
                scale = 0.38
            elif instr['size'] == 'small':
                scale = 0.25
            else:
                scale = 0.32  # Uniform default
            
            img_aspect = img_rgba.width / img_rgba.height
            max_size = int(min(safe_width, safe_height) * scale)
            
            if img_aspect > 1:
                new_width = max_size
                new_height = int(max_size / img_aspect)
            else:
                new_height = max_size
                new_width = int(max_size * img_aspect)
            
            img_resized = img_rgba.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_with_shadow = add_drop_shadow(img_resized, offset=(8, 8), blur_radius=12)
            
            processed_products.append({
                'image': img_with_shadow,
                'instr': instr,
                'index': i
            })
        
        # MINIMAL gaps - almost touching
        gap_x = int(target_width * 0.005)   # 0.5% horizontal gap
        gap_y = int(target_height * 0.005)  # 0.5% vertical gap
        
        # Calculate total width and height of grid
        row1_width = processed_products[0]['image'].width + gap_x + processed_products[1]['image'].width
        row2_width = processed_products[2]['image'].width + gap_x + processed_products[3]['image'].width
        max_grid_width = max(row1_width, row2_width)
        
        col1_height = processed_products[0]['image'].height + gap_y + processed_products[2]['image'].height
        col2_height = processed_products[1]['image'].height + gap_y + processed_products[3]['image'].height
        max_grid_height = max(col1_height, col2_height)
        
        # Center the grid within canvas
        grid_start_x = (target_width - max_grid_width) // 2
        grid_start_y = (target_height - max_grid_height) // 2
        
        # Position in 2x2 grid: top-left, top-right, bottom-left, bottom-right
        positions = [
            (grid_start_x, grid_start_y),  # Top-left
            (grid_start_x + processed_products[0]['image'].width + gap_x, grid_start_y),  # Top-right
            (grid_start_x, grid_start_y + processed_products[0]['image'].height + gap_y),  # Bottom-left
            (grid_start_x + processed_products[2]['image'].width + gap_x, grid_start_y + processed_products[1]['image'].height + gap_y),  # Bottom-right
        ]
        
        for idx, prod in enumerate(processed_products):
            img_with_shadow = prod['image']
            instr = prod['instr']
            
            # Check for specific position override
            if instr['position'] == 'center':
                x = (target_width - img_with_shadow.width) // 2
                y = (target_height - img_with_shadow.height) // 2
            else:
                # Use grid position
                x, y = positions[idx]
            
            composite.paste(img_with_shadow, (x, y), img_with_shadow)
            print(f"  Placed product {idx + 1} at ({x}, {y}), size {img_with_shadow.size}")
            print(f"  Placed product {idx + 1} at ({x}, {y})")
    
    else:
        # 5+ PRODUCTS: Tight clustered composition
        # Intentional grouping like reference images with safe zones
        
        print(f"📐 Layout: {num_products}-product tight cluster")
        
        # Safe zone: 85% of canvas to prevent cutoff
        safe_width = int(target_width * 0.85)
        safe_height = int(target_height * 0.85)
        safe_x = (target_width - safe_width) // 2
        safe_y = (target_height - safe_height) // 2
        
        # Identify hero products based on priority
        products_with_data = [(products_rgba[i], product_instructions[i], i) for i in range(len(products_rgba))]
        products_with_data.sort(key=lambda x: x[1]['priority'], reverse=True)
        
        # Top 1-2 priority products are heroes
        hero_indices = [products_with_data[i][2] for i in range(min(2, len(products_with_data))) 
                       if products_with_data[i][1]['priority'] >= 2]
        
        # If no high priority, feature first 1-2
        if not hero_indices:
            hero_indices = [0] if num_products == 5 else [0, 1]
        
        placed_rects = []
        
        # Place hero products first (larger, centered within safe zone)
        for idx in hero_indices[:2]:
            img_rgba, filename = products_rgba[idx]
            instr = product_instructions[idx]
            
            # Reduced hero size for tighter cluster
            if instr['size'] == 'large':
                scale = 0.40
            else:
                scale = 0.35
            
            img_aspect = img_rgba.width / img_rgba.height
            max_size = int(min(safe_width, safe_height) * scale)
            
            if img_aspect > 1:
                new_width = max_size
                new_height = int(max_size / img_aspect)
            else:
                new_height = max_size
                new_width = int(max_size * img_aspect)
            
            img_resized = img_rgba.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_with_shadow = add_drop_shadow(img_resized, offset=(8, 8), blur_radius=12)
            
            # Position hero centrally within safe zone
            if instr['position'] == 'center' or instr['position'] == 'auto':
                x = (target_width - img_with_shadow.width) // 2 + random.randint(-15, 15)
                y = (target_height - img_with_shadow.height) // 2 + random.randint(-15, 15)
            elif instr['position'] == 'left':
                x = safe_x + safe_width // 4 - img_with_shadow.width // 2
                y = (target_height - img_with_shadow.height) // 2 + random.randint(-15, 15)
            elif instr['position'] == 'right':
                x = safe_x + 3 * safe_width // 4 - img_with_shadow.width // 2
                y = (target_height - img_with_shadow.height) // 2 + random.randint(-15, 15)
            else:
                x = (target_width - img_with_shadow.width) // 2 + random.randint(-15, 15)
                y = (target_height - img_with_shadow.height) // 2 + random.randint(-15, 15)
            
            composite.paste(img_with_shadow, (x, y), img_with_shadow)
            placed_rects.append((x, y, x + img_with_shadow.width, y + img_with_shadow.height))
            print(f"  Placed HERO product {idx + 1} at ({x}, {y}), size {img_with_shadow.size}")
        
        # Place supporting products tightly clustered around heroes
        for idx, (img_rgba, filename) in enumerate(products_rgba):
            if idx in hero_indices[:2]:
                continue
            
            instr = product_instructions[idx]
            
            # Reduced supporting product size
            if instr['size'] == 'large':
                scale = 0.27
            elif instr['size'] == 'small':
                scale = 0.18
            else:
                scale = 0.22
            
            img_aspect = img_rgba.width / img_rgba.height
            max_size = int(min(safe_width, safe_height) * scale)
            
            if img_aspect > 1:
                new_width = max_size
                new_height = int(max_size / img_aspect)
            else:
                new_height = max_size
                new_width = int(max_size * img_aspect)
            
            img_resized = img_rgba.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_with_shadow = add_drop_shadow(img_resized, offset=(8, 8), blur_radius=12)
            
            # Find non-overlapping position tightly clustered near center within safe zone
            attempts = 0
            placed = False
            while attempts < 100 and not placed:
                # Tight cluster around center - within safe zone only
                cluster_x_min = safe_x + safe_width // 3
                cluster_x_max = safe_x + 2 * safe_width // 3 - img_with_shadow.width
                cluster_y_min = safe_y + safe_height // 3
                cluster_y_max = safe_y + 2 * safe_height // 3 - img_with_shadow.height
                
                x = random.randint(cluster_x_min, max(cluster_x_min, cluster_x_max))
                y = random.randint(cluster_y_min, max(cluster_y_min, cluster_y_max))
                
                new_rect = (x, y, x + img_with_shadow.width, y + img_with_shadow.height)
                
                # Check overlap with minimal tolerance (products very close)
                overlap = False
                for rect in placed_rects:
                    overlap_margin = 10  # Very tight clustering
                    if not (new_rect[2] < rect[0] - overlap_margin or 
                           new_rect[0] > rect[2] + overlap_margin or 
                           new_rect[3] < rect[1] - overlap_margin or 
                           new_rect[1] > rect[3] + overlap_margin):
                        overlap = True
                        break
                
                if not overlap:
                    composite.paste(img_with_shadow, (x, y), img_with_shadow)
                    placed_rects.append(new_rect)
                    placed = True
                    print(f"  Placed supporting product {idx + 1} at ({x}, {y}), size {img_with_shadow.size}")
                
                attempts += 1
            
            if not placed:
                # Fallback: place within safe zone
                fallback_x = random.randint(safe_x + 20, safe_x + safe_width - img_with_shadow.width - 20)
                fallback_y = random.randint(safe_y + 20, safe_y + safe_height - img_with_shadow.height - 20)
                composite.paste(img_with_shadow, (fallback_x, fallback_y), img_with_shadow)
                print(f"  Placed supporting product {idx + 1} at ({fallback_x}, {fallback_y}), size {img_with_shadow.size} [fallback]")
    
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

def export_file(img, mime, image_type=None):
    buf = io.BytesIO()
    
    if mime == "image/png":
        # For logos, STRICT <100KB enforcement
        if image_type == "brand_logo":
            # Try progressive compression levels
            for compress_level in range(6, 10):
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True, compress_level=compress_level)
                size_kb = len(buf.getvalue()) / 1024
                
                if size_kb < 100:
                    print(f"  ✅ Logo optimized to {size_kb:.1f}KB (compress_level={compress_level})")
                    return buf.getvalue()
            
            # If still too large, convert to palette mode and reduce colors
            print(f"  ⚠️ Logo still large, applying palette reduction...")
            
            # Try with 256 colors first
            for num_colors in [256, 192, 128, 96, 64]:
                buf = io.BytesIO()
                # Convert to palette mode with specified colors
                img_palette = img.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
                img_palette.save(buf, format="PNG", optimize=True, compress_level=9)
                size_kb = len(buf.getvalue()) / 1024
                
                if size_kb < 100:
                    print(f"  ✅ Logo optimized to {size_kb:.1f}KB ({num_colors} colors)")
                    return buf.getvalue()
            
            # Final attempt: maximum compression with minimal colors
            buf = io.BytesIO()
            img_palette = img.convert('P', palette=Image.ADAPTIVE, colors=64)
            img_palette.save(buf, format="PNG", optimize=True, compress_level=9)
            size_kb = len(buf.getvalue()) / 1024
            
            if size_kb >= 100:
                print(f"  ❌ WARNING: Logo is {size_kb:.1f}KB - EXCEEDS 100KB LIMIT!")
            else:
                print(f"  ✅ Logo compressed to {size_kb:.1f}KB")
            
            return buf.getvalue()
        else:
            img.save(buf, format="PNG")
    else:
        img.save(buf, format="JPEG", quality=92, optimize=True)
    
    return buf.getvalue()
