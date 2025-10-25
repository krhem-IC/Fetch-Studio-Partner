"""
Configuration file for Fetch Studio Imagery app.
Contains color palette, image specifications, and validation rules.
"""

# Fetch Brand Color Palette (RGB values)
FETCH_COLORS = {
    "Fetch Red": (255, 51, 68),
    "Fetch Blue": (41, 98, 255),
    "Fetch Green": (0, 209, 178),
    "Fetch Yellow": (255, 197, 0),
    "Fetch Purple": (138, 43, 226),
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
    "Light Gray": (240, 240, 240),
}

# Image specifications for each image type
IMAGE_SPECS = {
    "Offer Tile": {
        "width": 800,
        "height": 600,
        "format": ["PNG", "JPEG"],
        "max_file_size_mb": 2,
        "allowed_backgrounds": list(FETCH_COLORS.keys()),
    },
    "Offer Detail": {
        "width": 1200,
        "height": 800,
        "format": ["PNG", "JPEG"],
        "max_file_size_mb": 3,
        "allowed_backgrounds": list(FETCH_COLORS.keys()),
    },
    "Brand Hero": {
        "width": 1920,
        "height": 1080,
        "format": ["PNG", "JPEG"],
        "max_file_size_mb": 5,
        "allowed_backgrounds": list(FETCH_COLORS.keys()),
    },
    "Brand Logo": {
        "width": 512,
        "height": 512,
        "format": ["PNG"],
        "max_file_size_mb": 1,
        "allowed_backgrounds": ["White", "Transparent"],
    },
}

# Validation rules
VALIDATION_RULES = {
    "no_text_overlays": True,
    "no_gradients": True,
    "no_watermarks": True,
    "no_extra_logos": True,
    "must_use_fetch_colors": True,
}
