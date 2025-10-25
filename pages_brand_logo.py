import streamlit as st
from config import IMAGE_TYPES, APPROVED_SWATCHES
from utils import build_prompt, generate_image_stub, enforce_specs, validate_image, export_file, load_uploaded_image, suggest_filename

def render_brand_logo_page():
    st.header("Brand Logo")
    st.write("Upload an image to validate and process it for Fetch compliance.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'heic', 'heif'],
        help="Upload any image format - HEIC, PNG, JPG, GIF, etc.",
        key="brand_logo_upload"
    )
    
    if uploaded_file is not None:
        try:
            # Load and process the image
            img = load_uploaded_image(uploaded_file)
            
            # Show original image
            st.image(img, caption="Uploaded Image", use_column_width=True)
            
            # User inputs for context (used for prompt generation)
            col1, col2 = st.columns(2)
            with col1:
                brand = st.text_input("Brand")
            with col2:
                bg = st.selectbox("Expected Background Color", [f"{p['name']} {p['hex']}" for p in APPROVED_SWATCHES], key="upload_bg_logo")
            
            bg_hex = bg.split()[-1]
            
            # Process and validate
            img_processed = enforce_specs(img, "brand_logo")
            report = validate_image(img_processed, "brand_logo", bg_hex)
            
            # Generate prompt for reference
            if brand:
                prompt = build_prompt("brand_logo", brand, [], bg_hex, "")
                with st.expander("AI Prompt for this image type"):
                    st.code(prompt)
            
            st.subheader("Validation Results")
            for c in report["checks"]:
                status = "✅" if c["ok"] else "❌"
                st.write(f"{status} {c['name']}")
            
            # Show processed image if different from original
            if img_processed.size != img.size:
                st.subheader("Processed Image")
                st.image(img_processed, caption=f"Resized to {img_processed.size}", use_column_width=True)
            
            # Download options
            if report["pass"]:
                st.success("✅ Image passes all validation checks!")
                filename = suggest_filename("brand_logo", bg_hex, uploaded_file.name)
                data = export_file(img_processed, report["mime"])
                st.download_button("Download Processed Image", data=data, file_name=filename, mime=report["mime"])
            else:
                st.error("❌ Image failed validation. Please fix the issues and try again.")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    else:
        st.info("👆 Please upload an image to get started")

if __name__ == "__main__":
    render_brand_logo_page()
