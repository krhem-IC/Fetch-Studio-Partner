import streamlit as st
from config import IMAGE_TYPES, APPROVED_SWATCHES
from utils import load_uploaded_image, enforce_specs, validate_image, export_file, suggest_filename, build_prompt, create_composite_image

st.set_page_config(page_title="Fetch Studio Partner", layout="wide")

st.title("Fetch Studio Partner")
st.caption("Professional image validation and processing for Fetch brand compliance")

st.markdown("""
**How it works:**
1. 📤 Upload one or more images (any format supported)
2. 📝 Enter brand details and product organization notes  
3. 🎯 Choose image type and background color
4. 🚀 Create one final compliant image
5. 📥 Download your ready-to-use file
""")

st.markdown("---")

# Main workflow
st.header("📤 Upload Images")
uploaded_files = st.file_uploader(
    "Choose image files",
    type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'heic', 'heif'],
    accept_multiple_files=True,
    help="Upload one or more images - they will be combined into one final image"
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} image(s) uploaded successfully!")
    
    # Show uploaded images in a grid
    if len(uploaded_files) <= 3:
        cols = st.columns(len(uploaded_files))
    else:
        cols = st.columns(3)
    
    loaded_images = []
    for i, uploaded_file in enumerate(uploaded_files):
        with cols[i % len(cols)]:
            try:
                img = load_uploaded_image(uploaded_file)
                loaded_images.append((img, uploaded_file.name))
                st.image(img, caption=uploaded_file.name, use_column_width=True)
            except Exception as e:
                st.error(f"Could not load {uploaded_file.name}: {str(e)}")
    
    if loaded_images:
        st.header("📝 Project Information")
        
        col1, col2 = st.columns(2)
        with col1:
            brand_name = st.text_input("Brand Name", placeholder="e.g., Target, Walmart")
            product_callouts = st.text_area(
                "Product Organization & Call-outs", 
                placeholder="e.g., Place cereal box in center, rotate coffee bag to show front label, arrange produce in foreground",
                help="Describe how products should be positioned, rotated, or arranged in the final image"
            )
            
            # Example instructions
            st.caption("💡 **Examples of what you can specify:**")
            st.caption("• **Position**: 'center main product', 'place on left', 'arrange at bottom'")
            st.caption("• **Size**: 'make cereal box large', 'keep snacks small'") 
            st.caption("• **Rotation**: 'rotate coffee bag', 'turn label forward'")
            st.caption("• **Featured items**: 'feature breakfast items', 'highlight main product'")
        
        with col2:
            image_type = st.selectbox(
                "Final Image Type",
                options=list(IMAGE_TYPES.keys()),
                format_func=lambda x: IMAGE_TYPES[x]["label"]
            )
            
            background_color = st.selectbox(
                "Background Color",
                options=[f"{p['name']} {p['hex']}" for p in APPROVED_SWATCHES]
            )
        
        # Special handling for brand hero
        if image_type == "brand_hero":
            st.subheader("🎯 Brand Hero Options")
            hero_option = st.radio(
                "What do you need?",
                ["Compose from uploaded images", "Generate new lifestyle image"],
                help="Choose whether to create from your uploaded images or get guidance for AI generation"
            )
            
            if hero_option == "Generate new lifestyle image":
                lifestyle_direction = st.text_area(
                    "Lifestyle Design Direction",
                    placeholder="e.g., Modern kitchen setting with natural lighting, young family breakfast scene, outdoor picnic atmosphere",
                    help="Describe the mood, setting, and atmosphere for the new image"
                )
            else:
                lifestyle_direction = ""
        else:
            hero_option = "Compose from uploaded images"
            lifestyle_direction = ""
        
        if st.button("🚀 Create Final Image", type="primary"):
            bg_hex = background_color.split()[-1]
            
            st.header("🎨 Creating Your Image")
            
            # Generate prompt for reference if needed
            if image_type == "brand_hero" and hero_option == "Generate new lifestyle image":
                products = [p.strip() for p in product_callouts.split(",") if p.strip()] if product_callouts else []
                prompt = build_prompt(image_type, brand_name, products, bg_hex, lifestyle_direction)
                
                st.info("🤖 **AI Generation Mode** - Here's your custom prompt for AI image generation")
                with st.expander("🎨 AI Generation Prompt (Click to expand)"):
                    st.code(prompt)
                    st.markdown("""
                    **How to use this prompt:**
                    1. Copy the prompt above
                    2. Use it with AI tools like DALL-E, Midjourney, or Stable Diffusion
                    3. Upload the generated result back here for final processing
                    """)
                
                # Still create a composite from uploaded images as reference
                st.write("**Reference Composite from Your Uploads:**")
            
            # Create the final composite image
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Processing uploaded images...")
                progress_bar.progress(25)
                
                # Create composite image from all uploaded images
                final_image = create_composite_image(loaded_images, image_type, bg_hex, product_callouts)
                
                status_text.text("Applying specifications...")
                progress_bar.progress(50)
                
                # Enforce specs (resize to correct dimensions)
                final_image = enforce_specs(final_image, image_type)
                
                status_text.text("Validating compliance...")
                progress_bar.progress(75)
                
                # Validate the final image
                report = validate_image(final_image, image_type, bg_hex)
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📸 Your Final Image")
                    st.image(final_image, use_column_width=True)
                    
                    # Show specs
                    spec = IMAGE_TYPES[image_type]
                    st.info(f"**Specifications:** {spec['size'][0]}×{spec['size'][1]} {spec['format'].upper()}")
                
                with col2:
                    st.subheader("✅ Validation Results")
                    passed_checks = 0
                    total_checks = len(report["checks"])
                    
                    for check in report["checks"]:
                        if check["ok"]:
                            st.success(f"✅ {check['name']}")
                            passed_checks += 1
                        else:
                            st.error(f"❌ {check['name']}")
                    
                    # Show validation summary
                    if passed_checks == total_checks:
                        st.success(f"🎉 Perfect! All {total_checks} validation checks passed!")
                    else:
                        st.warning(f"⚠️ {passed_checks}/{total_checks} checks passed.")
                
                # Download section
                st.subheader("📥 Download Your Image")
                if passed_checks == total_checks:
                    filename = suggest_filename(image_type, bg_hex, f"{brand_name}_final" if brand_name else "final")
                    data = export_file(final_image, report["mime"])
                    st.download_button(
                        f"📥 Download {filename}",
                        data=data,
                        file_name=filename,
                        mime=report["mime"],
                        use_container_width=True
                    )
                    st.success("Your image is ready for use! 🚀")
                else:
                    st.info("💡 Some validation checks failed. The image has been created but may need adjustments before use.")
                    filename = suggest_filename(image_type, bg_hex, f"{brand_name}_draft" if brand_name else "draft")
                    data = export_file(final_image, report["mime"])
                    st.download_button(
                        f"📥 Download Draft {filename}",
                        data=data,
                        file_name=filename,
                        mime=report["mime"],
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"Error creating final image: {str(e)}")
                st.error("Please check your uploaded images and try again.")

else:
    # Show helpful information when no files uploaded
    st.info("👆 Upload one or more images to get started")
    
    with st.sidebar:
        st.header("📋 Quick Reference")
        
        st.subheader("Image Types")
        for key, spec in IMAGE_TYPES.items():
            st.write(f"**{spec['label']}**")
            st.write(f"Size: {spec['size'][0]}×{spec['size'][1]}")
            st.write(f"Format: {spec['format'].upper()}")
            st.write("")
        
        st.subheader("Approved Colors")
        cols = st.columns(2)
        for i, swatch in enumerate(APPROVED_SWATCHES):
            with cols[i % 2]:
                st.color_picker(
                    swatch["name"], 
                    swatch["hex"], 
                    key=f"ref_color_{i}", 
                    disabled=True
                )
