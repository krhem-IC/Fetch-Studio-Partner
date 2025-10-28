import streamlit as st
from PIL import Image
from config import IMAGE_TYPES, APPROVED_SWATCHES
from utils import load_uploaded_image, enforce_specs, validate_image, export_file, suggest_filename, build_prompt, create_composite_image

st.set_page_config(page_title="Fetch Studio Partner", page_icon="🎨", layout="wide")

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

# STEP 1: Choose image type FIRST
st.header("🎯 Step 1: Choose Image Type")
image_type = st.selectbox(
    "What type of image do you need?",
    options=list(IMAGE_TYPES.keys()),
    format_func=lambda x: IMAGE_TYPES[x]["label"],
    help="Select the image type first - this determines upload limits and requirements"
)

# Show specs for selected type
spec = IMAGE_TYPES[image_type]
st.info(f"📐 **{spec['label']}**: {spec['size'][0]}×{spec['size'][1]} {spec['format'].upper()} (Max: 250KB)")

# Special handling for Brand Hero - choose option FIRST
brand_hero_option = None
if image_type == "brand_hero":
    st.subheader("🎨 Brand Hero Options")
    brand_hero_option = st.radio(
        "How would you like to create your Brand Hero?",
        [
            "📤 Resize uploaded image to 1200x857",
            "🎨 Generate AI lifestyle image"
        ],
        help="Choose to resize your own image or generate a new lifestyle image with AI"
    )
    
    if "Resize" in brand_hero_option:
        max_products = 1
        st.caption(f"✅ Upload 1 image to resize to {spec['size'][0]}×{spec['size'][1]}")
    else:
        max_products = 5  # Allow product uploads for AI generation
        st.caption("✨ Upload 1-5 product images - AI will generate a lifestyle scene featuring your products")
else:
    # Set max uploads based on image type
    if image_type in ["offer_tile", "offer_detail"]:
        max_products = 7
        st.caption(f"✅ You can upload up to {max_products} product images for {spec['label']}")
    else:  # brand_logo
        max_products = 1
        st.caption(f"ℹ️ {spec['label']} requires exactly 1 logo image")

st.markdown("---")

# STEP 2: Upload Images
st.header(f"📤 Step 2: Upload Your Images")
uploaded_files = st.file_uploader(
    f"Choose image files (Max: {max_products})",
    type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'heic', 'heif'],
    accept_multiple_files=True,
    help=f"Upload up to {max_products} images - they will be combined into one final {spec['label']}"
)

# Enforce max limit
if uploaded_files and len(uploaded_files) > max_products:
    st.error(f"❌ Too many files! Please upload maximum {max_products} images for {spec['label']}")
    uploaded_files = uploaded_files[:max_products]
    st.warning(f"⚠️ Only using the first {max_products} files")

# Initialize loaded_images
loaded_images = []

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} image(s) uploaded successfully!")
    
    # Show uploaded images in a grid
    if len(uploaded_files) <= 3:
        cols = st.columns(len(uploaded_files))
    else:
        cols = st.columns(3)
    
    for i, uploaded_file in enumerate(uploaded_files):
        with cols[i % len(cols)]:
            try:
                img = load_uploaded_image(uploaded_file)
                loaded_images.append((img, uploaded_file.name))
                st.image(img, caption=uploaded_file.name, use_column_width=True)
            except Exception as e:
                st.error(f"Could not load {uploaded_file.name}: {str(e)}")

# Handle AI-generated Brand Hero (with product uploads)
if image_type == "brand_hero" and brand_hero_option and "Generate AI" in brand_hero_option and loaded_images:
    st.header("📝 Step 3: Describe Your Brand Hero Scene")
    
    # Show uploaded products
    st.success(f"✅ {len(loaded_images)} product image(s) will be featured in your AI-generated scene")
    
    # Show example reference
    with st.expander("💡 View Brand Hero Examples for Inspiration"):
        st.markdown("""
        **Great Brand Hero scenes include:**
        - 🏠 **Kitchen/Home Setting**: Products on counter with natural lighting (e.g., McCafe on kitchen counter)
        - 🌳 **Outdoor Scene**: Picnic table with checkered cloth and fresh setting (e.g., Snapple outdoor scene)
        - 🍓 **Ingredient Focused**: Product with fresh fruits/ingredients on wooden board (e.g., Celsius with strawberries)
        - ✨ **Clean & Modern**: Dark counter with ice cream products and spoons (e.g., So Delicious)
        - 🪟 **Natural Light**: Windowsill with plants and soft morning light (e.g., Happy Baby)
        - 🎨 **Bold & Graphic**: Product-forward with vibrant brand colors (e.g., Hi-C splash)
        
        **Key Elements:**
        - Lifestyle context (not just product on white)
        - Natural or warm lighting
        - Props that complement the brand story
        - Authentic, real-life feel
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        brand_name = st.text_input(
            "Brand Name *", 
            placeholder="e.g., Coca-Cola, Snapple, McCafe",
            help="Enter the brand name for your lifestyle image"
        )
        
        lifestyle_description = st.text_area(
            "Describe Your Lifestyle Scene *",
            placeholder=f"""Your {len(loaded_images)} uploaded product(s) will be featured in the scene. Describe the lifestyle setting:

Example: "Modern kitchen counter with my coffee products and colorful mugs, soft natural window light from the left, warm and cozy morning atmosphere, wooden cutting board in background"

Or: "Outdoor picnic scene with blue checkered tablecloth, my tea bottles with fresh lemons, bright sunny day, refreshing summer vibe"

Or: "Dark granite counter with my products, scattered berries and waffle pieces, moody studio lighting, indulgent dessert moment"

Be specific about: setting, lighting, mood, props, atmosphere (your products will be automatically included)""",
            help="Describe the lifestyle setting - your uploaded products will be featured in the scene",
            height=200
        )
    
    with col2:
        background_color = st.selectbox(
            "Primary Color Theme",
            options=[f"{p['name']} {p['hex']}" for p in APPROVED_SWATCHES],
            help="Choose a color that complements your brand - this guides the overall color palette"
        )
        
        bg_hex = background_color.split()[-1]
        st.markdown(f"""
        <div style="background-color: {bg_hex}; padding: 20px; border-radius: 8px; text-align: center; margin-top: 10px;">
            <span style="color: {'#000000' if bg_hex in ['#FFFFFF', '#F9DC5C'] else '#FFFFFF'}; font-weight: bold;">
                Color Theme Preview
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"📐 Creating: **Brand Hero**")
        st.caption(f"Size: 1200×857 (landscape)")
        st.caption(f"Format: JPG (Max: 250KB)")
        st.caption("")
        st.caption("💡 **Tip**: Be as specific as possible about the setting, lighting, and mood you want!")
    
    if st.button("🎨 Generate AI Prompt", type="primary"):
        if not brand_name or not lifestyle_description:
            st.error("❌ Please provide both brand name and scene description")
        else:
            st.header("✨ Your AI Brand Hero Prompt")
            
            try:
                # Create product description from uploaded images
                product_names = [filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ') for _, filename in loaded_images]
                
                # Generate AI prompt with product names
                prompt = build_prompt(image_type, brand_name, product_names, bg_hex, lifestyle_description)
                
                st.success("🎨 **AI Prompt Generated Successfully!**")
                
                # Show product list
                st.info(f"**Products to feature:** {', '.join(product_names)}")
                
                # Show the prompt in a nice format
                st.markdown("### 📋 Copy this prompt to your AI tool:")
                st.code(prompt, language="text")
                
                # Instructions
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🤖 How to Use This Prompt:")
                    st.markdown("""
                    1. **Copy the prompt** above (click the copy icon)
                    2. **Choose your AI tool**:
                       - DALL-E 3 (ChatGPT Plus)
                       - Midjourney
                       - Stable Diffusion
                       - Adobe Firefly
                    3. **Paste the prompt** into the AI tool
                    4. **Generate** at 1200×857 or larger
                    5. **Download** your AI-generated image
                    """)
                
                with col2:
                    st.markdown("### ⬅️ Next Steps:")
                    st.markdown("""
                    After generating your image:
                    
                    1. Come back to this app
                    2. Select **"📤 Resize uploaded image to 1200x857"**
                    3. Upload your AI-generated lifestyle image
                    4. We'll resize it to exact Fetch specs
                    5. Download your final Brand Hero (<250KB)
                    
                    💡 **Note**: The AI-generated image should feature your uploaded products in the lifestyle scene!
                    """)
                    
                    st.info("💡 Save this prompt! You can refine and regenerate as needed.")
                
            except Exception as e:
                st.error(f"Error generating prompt: {str(e)}")

elif loaded_images:
        st.header("📝 Step 3: Project Details")
        
        # Special handling for brand logo - ask first
        logo_option = None
        if image_type == "brand_logo":
            st.subheader("🎨 Brand Logo Options")
            logo_option = st.radio(
                "What does your logo need?",
                [
                    "Logo already has colored background - just resize",
                    "Add white background to logo",
                    "Add colored background to logo"
                ],
                help="Choose the right option for your logo: resize existing, add white, or add custom color"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            brand_name = st.text_input("Brand Name", placeholder="e.g., Target, Walmart")
            product_callouts = st.text_area(
                "Product Organization & Call-outs (Optional)", 
                placeholder="Leave blank for automatic professional layouts, or add specific instructions...\n\nExamples:\n• 'Put the 2 liter bottle in the middle'\n• 'Make the gift set large and centered'\n• 'First product on left, second in center'\n• 'Keep the multipack in the background'",
                help="FLEXIBLE: Leave blank for smart defaults, or be as specific as you want!"
            )
            
            # Example instructions with flexibility emphasis
            st.caption("✨ **This field is completely flexible:**")
            st.caption("• 🎯 **Blank**: Professional default layouts (fast & clean)")
            st.caption("• 📝 **Simple**: 'make it large', 'center it', 'on the left'")
            st.caption("• 🎨 **Specific**: 'put the 2-liter in middle, others small'")
            st.caption("• 🔧 **Detailed**: Full control over every product")
        
        with col2:
            # Conditional background color based on logo option
            if image_type == "brand_logo":
                if logo_option and "just resize" in logo_option:
                    # No color selection for resize-only
                    background_color = "White #FFFFFF"  # Default, won't be used
                    st.info("✅ Logo already has background - no color selection needed")
                elif logo_option and "white background" in logo_option:
                    # Fixed white for white background option
                    background_color = "White #FFFFFF"
                    st.info("⬜ Adding white background to logo")
                else:
                    # Show full color palette for colored background option
                    st.markdown("🎨 **Choose Background Color:**")
                    background_color = st.selectbox(
                        "Background Color",
                        options=[f"{p['name']} {p['hex']}" for p in APPROVED_SWATCHES] + 
                                ["Red #FF0000", "Blue #0000FF", "Green #00FF00", 
                                 "Yellow #FFFF00", "Orange #FFA500", "Purple #800080",
                                 "Black #000000"],
                        label_visibility="collapsed"
                    )
                    
                    # Show color preview
                    bg_hex = background_color.split()[-1]
                    st.markdown(f"""
                    <div style="background-color: {bg_hex}; padding: 20px; border-radius: 8px; text-align: center; margin-top: 10px;">
                        <span style="color: {'#000000' if bg_hex in ['#FFFFFF', '#F9DC5C', '#FFFF00', '#FFA500'] else '#FFFFFF'}; font-weight: bold;">
                            Background Preview
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Non-logo image types
                background_color = st.selectbox(
                    "Background Color",
                    options=[f"{p['name']} {p['hex']}" for p in APPROVED_SWATCHES]
                )
                
                # Show color preview for relevant image types
                if image_type in ["offer_tile", "offer_detail"]:
                    bg_hex = background_color.split()[-1]
                    st.markdown(f"""
                    <div style="background-color: {bg_hex}; padding: 20px; border-radius: 8px; text-align: center; margin-top: 10px;">
                        <span style="color: {'#000000' if bg_hex in ['#FFFFFF', '#F9DC5C'] else '#FFFFFF'}; font-weight: bold;">
                            Background Preview
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.write("")  # Spacing
            st.info(f"📐 Creating: **{IMAGE_TYPES[image_type]['label']}**")
            st.caption(f"Size: {IMAGE_TYPES[image_type]['size'][0]}×{IMAGE_TYPES[image_type]['size'][1]}")
            st.caption(f"Format: {IMAGE_TYPES[image_type]['format'].upper()}")
            st.caption(f"Max Size: {'100KB' if image_type == 'brand_logo' else '250KB'}")
            st.caption(f"Images: {len(loaded_images)}/{max_products}")
        
        if st.button("🚀 Create Final Image", type="primary"):
            bg_hex = background_color.split()[-1]
            
            st.header("🎨 Creating Your Image")
            
            # Create the final image
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Processing image...")
                progress_bar.progress(25)
                
                # Special handling for Brand Hero resize option
                if image_type == "brand_hero" and brand_hero_option and "Resize" in brand_hero_option:
                    # Simple resize to 1200x857
                    status_text.text("Resizing to 1200×857...")
                    img, filename = loaded_images[0]
                    final_image = img.resize((1200, 857), Image.Resampling.LANCZOS)
                    final_image = final_image.convert('RGB')
                else:
                    # Create composite image from uploaded images
                    final_image = create_composite_image(loaded_images, image_type, bg_hex, product_callouts, logo_option)
                
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
                    data = export_file(final_image, report["mime"], image_type)
                    st.download_button(
                        f"📥 Download {filename}",
                        data=data,
                        file_name=filename,
                        mime=report["mime"],
                        use_container_width=True
                    )
                    
                    # Show file size with STRICT enforcement
                    if image_type == "brand_logo":
                        size_kb = len(data) / 1024
                        if size_kb < 100:
                            st.success(f"✅ Your logo is ready for use! ({size_kb:.1f}KB / 100KB limit)")
                        else:
                            st.error(f"❌ LOGO TOO LARGE: {size_kb:.1f}KB exceeds 100KB limit!")
                            st.error("This logo cannot be used. Please simplify the design or use a different background color.")
                    else:
                        # All other images: 250KB limit
                        size_kb = len(data) / 1024
                        if size_kb < 250:
                            st.success(f"✅ Your image is ready for use! ({size_kb:.1f}KB / 250KB limit)")
                        else:
                            st.error(f"❌ IMAGE TOO LARGE: {size_kb:.1f}KB exceeds 250KB limit!")
                            st.error("This image cannot be used. Please try with a simpler image or lower quality.")
                else:
                    st.info("💡 Some validation checks failed. The image has been created but may need adjustments before use.")
                    filename = suggest_filename(image_type, bg_hex, f"{brand_name}_draft" if brand_name else "draft")
                    data = export_file(final_image, report["mime"], image_type)
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
