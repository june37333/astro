import streamlit as st
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.io import MemoryFile
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

# ---------------------------------
# Ensure data directory exists
# ---------------------------------
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

# ---------------------------------
# Streamlit App Configuration
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

# File upload or directory selection
use_upload = st.sidebar.checkbox("Use File Uploader", value=False)
if use_upload:
    uploaded = st.sidebar.file_uploader(
        "Upload CRISM Image (.img/.tif/.tiff)",
        type=['img', 'tif', 'tiff']
    )
    if not uploaded:
        st.warning("Please upload a .img, .tif, or .tiff file.")
        st.stop()
    # Handle uploaded file based on extension
    _, ext = os.path.splitext(uploaded.name)
    ext = ext.lower()
    if ext == '.img':
        # Write to temp file for rasterio to open
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.img')
        tmp.write(uploaded.getbuffer())
        tmp.flush()
        try:
            src = rasterio.open(tmp.name)
        except Exception as e:
            st.error(f"Failed to open uploaded .img file: {e}")
            st.stop()
    else:
        # Use MemoryFile for GeoTIFF
        memfile = MemoryFile(uploaded.read())
        try:
            src = memfile.open()
        except Exception as e:
            st.error(f"Failed to open uploaded file: {e}")
            st.stop()
else:
    # Directory selection
    files = [f for f in os.listdir(data_dir)
             if f.lower().endswith(('.img', '.tif', '.tiff'))]
    if not files:
        st.warning(
            f"No .img/.tif/.tiff files found in '{data_dir}'.\n"
            "Please upload via uploader or place file in data/."
        )
        st.stop()
    selected = st.sidebar.selectbox("Select CRISM File", files)
    file_path = os.path.join(data_dir, selected)
    try:
        src = rasterio.open(file_path)
    except Exception as e:
        st.error(f"Failed to open file from data/: {e}")
        st.stop()

# ---------------------------------
# Threshold sliders
# ---------------------------------
water_thresh = st.sidebar.slider(
    "Water Ratio Threshold (NIR/SWIR)", 0.0, 2.0, 0.6, 0.01
)
salt_thresh = st.sidebar.slider(
    "Salt Ratio Threshold (SWIR/Blue)", 0.0, 2.0, 0.5, 0.01
)
w_water = st.sidebar.slider("Water Score Weight", 0.0, 1.0, 0.6, 0.05)
w_salt = 1.0 - w_water
alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, 0.5, 0.05)

# ---------------------------------
# Load and Validate Image
# ---------------------------------
try:
    img = reshape_as_image(src.read())
finally:
    src.close()

# Ensure at least 5 bands
if img.ndim != 3 or img.shape[2] < 5:
    st.error("Image must have at least 5 bands: B, G, R, NIR, SWIR.")
    st.stop()

# ---------------------------------
# Compute Indices
# ---------------------------------
blue = img[:, :, 2].astype(float)
nir  = img[:, :, 3].astype(float)
swir = img[:, :, 4].astype(float)

def safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b>0)

water_ratio = safe_div(nir, swir)
salt_ratio  = safe_div(swir, blue)

# Habitability score [0-100]
habit_score = (water_ratio > water_thresh).astype(float) * w_water + \
              (salt_ratio < salt_thresh).astype(float) * w_salt
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

st.subheader("Overlay: Original + Habitability")
fig2, ax2 = plt.subplots(figsize=(8,8))
ax2.imshow(img[:, :, :3])
ax2.imshow(habit_pct, cmap='viridis', alpha=alpha, vmin=0, vmax=100)
ax2.axis('off')
st.pyplot(fig2)

# ---------------------------------
# Statistics
# ---------------------------------
st.subheader("Habitability Statistics")
avg = habit_pct.mean()
st.metric("Average Habitability", f"{avg:.2f}%")
min_v = int(habit_pct.min())
max_v = int(habit_pct.max())
st.write(f"Min: {min_v}%, Max: {max_v}%")

# ---------------------------------
# Footer: GitHub & Deployment Guide
# ---------------------------------
st.markdown("---")
st.markdown("""
**GitHub Setup & Deployment:**  
1. Add `app.py` and `requirements.txt` to your repo root.  
2. Place CRISM .img/.tif files in `data/` or upload via uploader.  
3. Commit & push to GitHub.  
4. Deploy via Streamlit Cloud.
""")

# ---------------------------------
# requirements.txt:
# streamlit
# rasterio
# numpy
# matplotlib
# ---------------------------------
