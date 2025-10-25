# Fetch Studio Imagery

A Streamlit app to help users generate, validate, and download Fetch-compliant studio images.

## Features

- **Generate AI Prompts**: Create prompts for image generation tools with Fetch brand guidelines
- **Validate Images**: Check if images meet size, format, and brand requirements
- **Create Blank Templates**: Generate blank images with correct specifications
- **Download Images**: Export validated images in the correct format

## Image Types Supported

1. **Offer Tile** (800x600px) - Product offers and promotions
2. **Offer Detail** (1200x800px) - Detailed offer pages
3. **Brand Hero** (1920x1080px) - Hero banners and featured content
4. **Brand Logo** (512x512px) - Brand logos and icons

## Installation

1. Clone the repository:
```bash
git clone https://github.com/krhem-IC/Fetch-Studio-Partner.git
cd Fetch-Studio-Partner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Project Structure

```
Fetch-Studio-Partner/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration (colors, specs, rules)
├── utils.py                  # Validation and utility functions
├── pages_offer_tile.py       # Offer Tile page
├── pages_offer_detail.py     # Offer Detail page
├── pages_brand_hero.py       # Brand Hero page
├── pages_brand_logo.py       # Brand Logo page
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Brand Guidelines

All images must follow these Fetch brand guidelines:

- ✅ Use only approved Fetch brand colors
- ✅ Solid color backgrounds only (no gradients)
- ❌ No text overlays or typography
- ❌ No watermarks or stamps
- ❌ No extra logos or branding elements

## Fetch Color Palette

- Fetch Red: RGB(255, 51, 68)
- Fetch Blue: RGB(41, 98, 255)
- Fetch Green: RGB(0, 209, 178)
- Fetch Yellow: RGB(255, 197, 0)
- Fetch Purple: RGB(138, 43, 226)
- White: RGB(255, 255, 255)
- Black: RGB(0, 0, 0)
- Light Gray: RGB(240, 240, 240)

## Validation

The app includes validation for:

- ✅ Image dimensions (width x height)
- ✅ File format (PNG/JPEG)
- ✅ File size limits
- ⚠️ Background color (placeholder implementation)
- ⚠️ Text overlay detection (placeholder implementation)
- ⚠️ Gradient detection (placeholder implementation)
- ⚠️ Watermark detection (placeholder implementation)
- ⚠️ Extra logo detection (placeholder implementation)

**Note**: Some validation checks marked with ⚠️ are placeholder implementations and would be enhanced with AI/ML capabilities in production.

## License

MIT License
