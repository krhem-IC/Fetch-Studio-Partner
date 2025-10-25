"""
Fetch Studio Imagery - Main Streamlit Application

A Streamlit app to help users generate, validate, and download 
Fetch-compliant studio images.
"""

import streamlit as st
from pages_offer_tile import render_offer_tile_page
from pages_offer_detail import render_offer_detail_page
from pages_brand_hero import render_brand_hero_page
from pages_brand_logo import render_brand_logo_page
from config import IMAGE_SPECS, FETCH_COLORS, VALIDATION_RULES


# Page configuration
st.set_page_config(
    page_title="Fetch Studio Imagery",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with navigation and information."""
    with st.sidebar:
        st.title("🎨 Fetch Studio Imagery")
        st.markdown("---")
        
        # Navigation
        st.header("Navigation")
        page = st.radio(
            "Select Image Type",
            options=[
                "🏠 Home",
                "📦 Offer Tile",
                "📄 Offer Detail",
                "🎯 Brand Hero",
                "🏷️ Brand Logo"
            ],
            key="navigation"
        )
        
        st.markdown("---")
        
        # Color palette reference
        st.header("Fetch Color Palette")
        for color_name, rgb in FETCH_COLORS.items():
            # Skip Transparent in the sidebar color palette display
            if color_name == "Transparent":
                continue
            st.markdown(
                f'<div style="display: flex; align-items: center; margin-bottom: 0.5rem;">'
                f'<div style="width: 30px; height: 30px; background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); '
                f'border: 1px solid #ccc; border-radius: 3px; margin-right: 10px;"></div>'
                f'<span>{color_name}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Validation rules
        st.header("Validation Rules")
        for rule, enabled in VALIDATION_RULES.items():
            if enabled:
                rule_text = rule.replace("_", " ").title()
                st.markdown(f"✅ {rule_text}")
        
        return page


def render_home_page():
    """Render the home page."""
    st.title("🎨 Fetch Studio Imagery")
    
    st.markdown("""
    Welcome to the Fetch Studio Imagery app! This tool helps you generate, validate, 
    and download Fetch-compliant studio images for various use cases.
    
    ## Features
    
    - **Generate AI Prompts**: Create prompts for image generation tools
    - **Validate Images**: Check if images meet Fetch brand requirements
    - **Create Blank Templates**: Generate blank images with correct specifications
    - **Download Images**: Export validated images in the correct format
    
    ## Image Types
    
    Select an image type from the sidebar to get started:
    """)
    
    # Display image specifications in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Offer Tile")
        st.markdown(f"""
        - **Size**: {IMAGE_SPECS['Offer Tile']['width']}x{IMAGE_SPECS['Offer Tile']['height']} pixels
        - **Format**: {', '.join(IMAGE_SPECS['Offer Tile']['format'])}
        - **Max Size**: {IMAGE_SPECS['Offer Tile']['max_file_size_mb']} MB
        - **Use Case**: Product offers and promotions
        """)
        
        st.subheader("🎯 Brand Hero")
        st.markdown(f"""
        - **Size**: {IMAGE_SPECS['Brand Hero']['width']}x{IMAGE_SPECS['Brand Hero']['height']} pixels
        - **Format**: {', '.join(IMAGE_SPECS['Brand Hero']['format'])}
        - **Max Size**: {IMAGE_SPECS['Brand Hero']['max_file_size_mb']} MB
        - **Use Case**: Hero banners and featured content
        """)
    
    with col2:
        st.subheader("📄 Offer Detail")
        st.markdown(f"""
        - **Size**: {IMAGE_SPECS['Offer Detail']['width']}x{IMAGE_SPECS['Offer Detail']['height']} pixels
        - **Format**: {', '.join(IMAGE_SPECS['Offer Detail']['format'])}
        - **Max Size**: {IMAGE_SPECS['Offer Detail']['max_file_size_mb']} MB
        - **Use Case**: Detailed offer pages
        """)
        
        st.subheader("🏷️ Brand Logo")
        st.markdown(f"""
        - **Size**: {IMAGE_SPECS['Brand Logo']['width']}x{IMAGE_SPECS['Brand Logo']['height']} pixels
        - **Format**: {', '.join(IMAGE_SPECS['Brand Logo']['format'])}
        - **Max Size**: {IMAGE_SPECS['Brand Logo']['max_file_size_mb']} MB
        - **Use Case**: Brand logos and icons
        """)
    
    st.markdown("---")
    
    st.subheader("📋 Brand Guidelines")
    
    st.markdown("""
    All images must follow these Fetch brand guidelines:
    
    - ✅ Use only approved Fetch brand colors
    - ✅ Solid color backgrounds only (no gradients)
    - ❌ No text overlays or typography
    - ❌ No watermarks or stamps
    - ❌ No extra logos or branding elements
    - ❌ No unauthorized color combinations
    
    **Note**: Some validation checks are placeholders and will be enhanced with AI/ML capabilities.
    """)
    
    st.markdown("---")
    
    st.info("👈 Select an image type from the sidebar to get started!")


def main():
    """Main application logic."""
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render the appropriate page
    if page == "🏠 Home":
        render_home_page()
    elif page == "📦 Offer Tile":
        render_offer_tile_page()
    elif page == "📄 Offer Detail":
        render_offer_detail_page()
    elif page == "🎯 Brand Hero":
        render_brand_hero_page()
    elif page == "🏷️ Brand Logo":
        render_brand_logo_page()


if __name__ == "__main__":
    main()
