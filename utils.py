"""
Utility functions for image validation and prompt building.
"""

from PIL import Image
import io
from typing import Tuple, Dict, List
from config import IMAGE_SPECS, VALIDATION_RULES, FETCH_COLORS


def validate_image_size(image: Image.Image, image_type: str) -> Tuple[bool, str]:
    """
    Validate that the image meets the size requirements for the given type.
    
    Args:
        image: PIL Image object
        image_type: Type of image (e.g., "Offer Tile", "Offer Detail")
    
    Returns:
        Tuple of (is_valid, message)
    """
    spec = IMAGE_SPECS[image_type]
    expected_width = spec["width"]
    expected_height = spec["height"]
    actual_width, actual_height = image.size
    
    if actual_width == expected_width and actual_height == expected_height:
        return True, f"✅ Image size is correct: {actual_width}x{actual_height}"
    else:
        return False, f"❌ Image size is {actual_width}x{actual_height}, but should be {expected_width}x{expected_height}"


def validate_image_format(image: Image.Image, image_type: str) -> Tuple[bool, str]:
    """
    Validate that the image format is allowed for the given type.
    
    Args:
        image: PIL Image object
        image_type: Type of image
    
    Returns:
        Tuple of (is_valid, message)
    """
    spec = IMAGE_SPECS[image_type]
    allowed_formats = spec["format"]
    actual_format = image.format
    
    if actual_format in allowed_formats:
        return True, f"✅ Image format is valid: {actual_format}"
    else:
        return False, f"❌ Image format is {actual_format}, but must be one of: {', '.join(allowed_formats)}"


def validate_file_size(file_bytes: bytes, image_type: str) -> Tuple[bool, str]:
    """
    Validate that the file size is within limits.
    
    Args:
        file_bytes: Image file as bytes
        image_type: Type of image
    
    Returns:
        Tuple of (is_valid, message)
    """
    spec = IMAGE_SPECS[image_type]
    max_size_mb = spec["max_file_size_mb"]
    actual_size_mb = len(file_bytes) / (1024 * 1024)
    
    if actual_size_mb <= max_size_mb:
        return True, f"✅ File size is valid: {actual_size_mb:.2f} MB"
    else:
        return False, f"❌ File size is {actual_size_mb:.2f} MB, but must be under {max_size_mb} MB"


def validate_background_color(image: Image.Image, expected_color: str) -> Tuple[bool, str]:
    """
    Placeholder validation for background color.
    In a real implementation, this would analyze the image pixels.
    
    Args:
        image: PIL Image object
        expected_color: Expected background color name
    
    Returns:
        Tuple of (is_valid, message)
    """
    # Placeholder: In production, would analyze actual pixel colors
    return True, f"⚠️ Background color validation (placeholder): Expected {expected_color}"


def validate_no_text_overlay(image: Image.Image) -> Tuple[bool, str]:
    """
    Placeholder validation for text overlays.
    In a real implementation, this would use OCR or ML to detect text.
    
    Args:
        image: PIL Image object
    
    Returns:
        Tuple of (is_valid, message)
    """
    # Placeholder: In production, would use OCR or computer vision
    return True, "⚠️ Text overlay check (placeholder): No text detected"


def validate_no_gradient(image: Image.Image) -> Tuple[bool, str]:
    """
    Placeholder validation for gradients.
    In a real implementation, this would analyze color transitions.
    
    Args:
        image: PIL Image object
    
    Returns:
        Tuple of (is_valid, message)
    """
    # Placeholder: In production, would analyze color gradients
    return True, "⚠️ Gradient check (placeholder): No gradients detected"


def validate_no_watermark(image: Image.Image) -> Tuple[bool, str]:
    """
    Placeholder validation for watermarks.
    In a real implementation, this would detect watermarks.
    
    Args:
        image: PIL Image object
    
    Returns:
        Tuple of (is_valid, message)
    """
    # Placeholder: In production, would detect watermarks
    return True, "⚠️ Watermark check (placeholder): No watermarks detected"


def validate_no_extra_logos(image: Image.Image) -> Tuple[bool, str]:
    """
    Placeholder validation for extra logos.
    In a real implementation, this would use object detection.
    
    Args:
        image: PIL Image object
    
    Returns:
        Tuple of (is_valid, message)
    """
    # Placeholder: In production, would use object detection
    return True, "⚠️ Logo check (placeholder): No extra logos detected"


def validate_image(image: Image.Image, file_bytes: bytes, image_type: str, background_color: str) -> List[str]:
    """
    Perform all validations on an image.
    
    Args:
        image: PIL Image object
        file_bytes: Image file as bytes
        image_type: Type of image
        background_color: Expected background color
    
    Returns:
        List of validation messages
    """
    messages = []
    
    # Size validation
    is_valid, msg = validate_image_size(image, image_type)
    messages.append(msg)
    
    # Format validation
    is_valid, msg = validate_image_format(image, image_type)
    messages.append(msg)
    
    # File size validation
    is_valid, msg = validate_file_size(file_bytes, image_type)
    messages.append(msg)
    
    # Background color validation
    is_valid, msg = validate_background_color(image, background_color)
    messages.append(msg)
    
    # Brand rules validation
    if VALIDATION_RULES["no_text_overlays"]:
        is_valid, msg = validate_no_text_overlay(image)
        messages.append(msg)
    
    if VALIDATION_RULES["no_gradients"]:
        is_valid, msg = validate_no_gradient(image)
        messages.append(msg)
    
    if VALIDATION_RULES["no_watermarks"]:
        is_valid, msg = validate_no_watermark(image)
        messages.append(msg)
    
    if VALIDATION_RULES["no_extra_logos"]:
        is_valid, msg = validate_no_extra_logos(image)
        messages.append(msg)
    
    return messages


def build_image_prompt(image_type: str, background_color: str, description: str) -> str:
    """
    Build a prompt for AI image generation.
    Placeholder implementation that constructs a basic prompt.
    
    Args:
        image_type: Type of image to generate
        background_color: Background color to use
        description: User's description of desired image
    
    Returns:
        Generated prompt string
    """
    spec = IMAGE_SPECS[image_type]
    width = spec["width"]
    height = spec["height"]
    
    # Get RGB value for the color
    rgb = FETCH_COLORS.get(background_color, (255, 255, 255))
    
    prompt = f"""Create a {image_type.lower()} image for Fetch Rewards.

Image specifications:
- Size: {width}x{height} pixels
- Background color: {background_color} (RGB: {rgb})
- Style: Clean, professional, brand-compliant

User description: {description}

Important requirements:
- Use solid {background_color} background only
- NO text overlays or typography
- NO gradients or color transitions
- NO watermarks
- NO extra logos
- Follow Fetch brand guidelines
- High quality, professional imagery
"""
    
    return prompt


def create_blank_image(image_type: str, background_color: str) -> Image.Image:
    """
    Create a blank image with the correct specifications.
    
    Args:
        image_type: Type of image to create
        background_color: Background color to use
    
    Returns:
        PIL Image object
    """
    spec = IMAGE_SPECS[image_type]
    width = spec["width"]
    height = spec["height"]
    
    # Get RGB value for the color
    rgb = FETCH_COLORS.get(background_color, (255, 255, 255))
    
    # Create image
    if background_color == "Transparent":
        image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    else:
        image = Image.new("RGB", (width, height), rgb)
    
    return image
