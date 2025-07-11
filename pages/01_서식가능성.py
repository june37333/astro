import streamlit as st
import rasterio
from rasterio.plot import reshape_as_image
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------
# App Configuration
# ---------------------------------
st.set_page_config(
    page_title="Mars Habitability Mapper",
    layout="wide"
)
st.title("🌌 Mars Surface Habitability Estimator")

# ---------------------------------
# Sidebar: Controls
# ---------------------------------
st.sidebar.header("🔧 Settings")

# File upload option
use_uploader = st.sidebar.checkbox("Use File Uploader", value=False)

data_dir = 'data'

if use_uploader:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CRISM Map-Projected Image (.img/.tif)",
        type=['img','tif','tiff']
    )
    if uploaded_file is None:
        st.warning("Please upload a CRISM image file to proceed.")
        st.stop()
    filepath = uploaded_file
else:
    # List available files in data_dir
    files = []
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.img','.tif','.tiff'))]
    if not files:
        st.warning(f"No .img or .tif files found in '{data_dir}' folder.")
        st.stop()
    selected = st.sidebar.selectbox("Select CRISM File", files)
    filepath = os.path.join(data_dir, selected)

# Threshold sliders
water_thresh = st.sidebar.slider(
    "Water Ratio Threshold (NIR/SWIR)", 0.0, 2.0, 0.6, 0.01
)
salt_thresh = st.sidebar.slider(
    "Salt Ratio Threshold (SWIR/Blue)", 0.0, 2.0, 0.5, 0.01
)
# Weights for combining scores
w_water = st.sidebar.slider("Water Score Weight", 0.0, 1.0, 0.6, 0.05)
w_salt = 1.0 - w_water

# ---------------------------------
# Load and Process Image
# ---------------------------------
with rasterio.open(filepath) as src:
    img = reshape_as_image(src.read())  # H x W x C

# Assume bands: [B, G, R, NIR, SWIR]
if img.shape[2] < 5:
    st.error("Image does not contain enough bands for NIR/SWIR analysis.")
    st.stop()

blue = img[:, :, 2].astype(float)
nir  = img[:, :, 3].astype(float)
swir = img[:, :, 4].astype(float)

# Compute indices
def safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b>0)

water_ratio = safe_div(nir, swir)
salt_ratio  = safe_div(swir, blue)

# Compute habitability score [0-1]
habit_score = (water_ratio > water_thresh).astype(float) * w_water + \
              (salt_ratio < salt_thresh).astype(float) * w_salt

# Scale to percentage
habit_pct = (habit_score * 100).astype(np.uint8)

# ---------------------------------
# Visualization
# ---------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original RGB Preview")
    st.image(img[:, :, :3], use_column_width=True)

with col2:
    st.subheader("Habitability Score Map")
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(habit_pct, cmap='viridis', vmin=0, vmax=100)
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Habitability (%)')
    st.pyplot(fig)

# Overlay on RGB
st.subheader("Overlay: Original + Habitability")
alpha = st.slider("Overlay Transparency", 0.0, 1.0, 0.5, 0.05)
fig2, ax2 = plt.subplots(figsize=(8,8))
ax2.imshow(img[:, :, :3])
ax2.imshow(habit_pct, cmap='viridis', alpha=alpha, vmin=0, vmax=100)
ax2.axis('off')
st.pyplot(fig2)

# ---------------------------------
# Statistics
# ---------------------------------
st.subheader("Habitability Statistics")
avg_score = habit_pct.mean()
st.metric("Average Habitability", f"{avg_score:.2f}%")
min_score = int(habit_pct.min())
max_score = int(habit_pct.max())
st.write(f"Min: {min_score}%, Max: {max_score}%")

# ---------------------------------
# Footer: GitHub & Deployment Guide
# ---------------------------------
st.markdown("---")
st.markdown("""
**GitHub Setup & Deployment:**  
1. Add `app.py` and `requirements.txt` to your repository root.  
2. Create a `data/` folder and place your CRISM `.img` or `.tif` files there.  
3. Commit and push your changes to GitHub.  
4. Link your repository in Streamlit Cloud for live deployment.
""")

# ---------------------------------
# requirements.txt content:
# streamlit
# rasterio
# numpy
# matplotlib
# ---------------------------------
