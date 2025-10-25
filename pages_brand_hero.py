"""
Brand Hero page for Fetch Studio Imagery app.
"""

import streamlit as st
from PIL import Image
import io
from utils import validate_image, build_image_prompt, create_blank_image
from config import IMAGE_SPECS, FETCH_COLORS


def render_brand_hero_page():
    """Render the Brand Hero page."""
    st.title("🎯 Brand Hero Image Generator")
    
    st.markdown("""
    Generate and validate Brand Hero images for Fetch Rewards.
    
    **Specifications:**
    - Size: 1920x1080 pixels (Full HD)
    - Format: PNG or JPEG
    - Max file size: 5 MB
    - Must use Fetch brand colors
    """)
    
    # Create tabs for different actions
    tab1, tab2, tab3 = st.tabs(["📝 Generate Prompt", "✅ Validate Image", "🎨 Create Blank"])
    
    with tab1:
        render_prompt_builder()
    
    with tab2:
        render_validator()
    
    with tab3:
        render_blank_creator()


def render_prompt_builder():
    """Render the prompt building section."""
    st.header("Generate Image Prompt")
    
    st.markdown("Create an AI-ready prompt for generating Brand Hero images.")
    
    # Background color selector
    background_color = st.selectbox(
        "Background Color",
        options=list(FETCH_COLORS.keys()),
        key="brand_hero_bg_color"
    )
    
    # Show color preview
    rgb = FETCH_COLORS[background_color]
    st.markdown(
        f'<div style="width: 100%; height: 50px; background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); '
        f'border: 1px solid #ccc; border-radius: 5px;"></div>',
        unsafe_allow_html=True
    )
    
    # Description input
    description = st.text_area(
        "Image Description",
        placeholder="Describe the image you want to generate (e.g., 'Dynamic lifestyle shot with shopping bags')",
        help="Provide a clear description of the desired image content",
        key="brand_hero_description"
    )
    
    # Generate prompt button
    if st.button("Generate Prompt", key="brand_hero_generate"):
        if description:
            prompt = build_image_prompt("Brand Hero", background_color, description)
            st.subheader("Generated Prompt")
            st.code(prompt, language="text")
            
            st.success("✅ Prompt generated! Copy this prompt to your image generation tool.")
        else:
            st.warning("Please enter an image description first.")


def render_validator():
    """Render the image validation section."""
    st.header("Validate Brand Hero Image")
    
    st.markdown("Upload an image to validate it meets Fetch requirements.")
    
    # Background color selector for validation
    background_color = st.selectbox(
        "Expected Background Color",
        options=list(FETCH_COLORS.keys()),
        key="brand_hero_validate_bg_color"
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"],
        help="Upload your Brand Hero image for validation",
        key="brand_hero_upload"
    )
    
    if uploaded_file is not None:
        # Read the file
        file_bytes = uploaded_file.read()
        
        try:
            # Open image
            image = Image.open(io.BytesIO(file_bytes))
            
            # Display image
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Validate
            st.subheader("Validation Results")
            messages = validate_image(image, file_bytes, "Brand Hero", background_color)
            
            for msg in messages:
                if "✅" in msg:
                    st.success(msg)
                elif "❌" in msg:
                    st.error(msg)
                else:
                    st.info(msg)
            
            # Download button
            st.subheader("Download Image")
            st.download_button(
                label="📥 Download Image",
                data=file_bytes,
                file_name=f"brand_hero_{background_color.lower().replace(' ', '_')}.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")


def render_blank_creator():
    """Render the blank image creator section."""
    st.header("Create Blank Brand Hero")
    
    st.markdown("Generate a blank Brand Hero image with the correct specifications.")
    
    # Background color selector
    background_color = st.selectbox(
        "Background Color",
        options=list(FETCH_COLORS.keys()),
        key="brand_hero_blank_bg_color"
    )
    
    # Show color preview
    rgb = FETCH_COLORS[background_color]
    st.markdown(
        f'<div style="width: 100%; height: 50px; background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); '
        f'border: 1px solid #ccc; border-radius: 5px;"></div>',
        unsafe_allow_html=True
    )
    
    if st.button("Create Blank Image", key="brand_hero_create_blank"):
        # Create blank image
        image = create_blank_image("Brand Hero", background_color)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # Display
        st.image(image, caption=f"Blank Brand Hero - {background_color}", use_container_width=True)
        
        # Download button
        st.download_button(
            label="📥 Download Blank Image",
            data=img_bytes,
            file_name=f"blank_brand_hero_{background_color.lower().replace(' ', '_')}.png",
            mime="image/png"
        )
