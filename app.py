import streamlit as st
from PIL import Image
import io, zipfile, cv2, numpy as np
from rembg import remove
from ultralytics import YOLO
from typing import Tuple, Optional

# ========== Configuration ==========
st.set_page_config(
    page_title="AI Photo Cropper Pro",
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="collapsed"
)

# ========== Model Loading ==========
@st.cache_resource
def load_yolo_model():
    try:
        return YOLO('yolov8n-seg.pt')
    except Exception as e:
        st.error(f"Failed to load YOLO model: {str(e)}")
        return None

# ========== Core Functions ==========
def enhanced_subject_detection(model, img: Image.Image) -> Optional[Tuple[int]]:
    """Detect main subject with combined YOLO and rembg approach"""
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # YOLO Segmentation
    if model:
        results = model.predict(img_cv, classes=0, verbose=False)
        for r in results:
            if r.masks is not None:
                masks = r.masks.xy
                if len(masks) > 0:
                    largest_mask = max(masks, key=lambda m: cv2.contourArea(m))
                    x,y,w,h = cv2.boundingRect(largest_mask.astype(np.int32))
                    return (x, y, x+w, y+h)

    # Fallback to rembg with expansion
    bg_removed = remove(img, post_process_mask=True)
    alpha = bg_removed.split()[-1]
    bbox = alpha.getbbox()
    
    if bbox:
        dx = int((bbox[2] - bbox[0]) * 0.05)
        dy = int((bbox[3] - bbox[1]) * 0.05)
        return (
            max(0, bbox[0] - dx),
            max(0, bbox[1] - dy),
            min(img.width, bbox[2] + dx),
            min(img.height, bbox[3] + dy)
        )
    return None

def smart_crop_to_target(img: Image.Image, bbox: Tuple[int], target_size: Tuple[int], zoom: float):
    """Crop image around subject to exactly match target dimensions"""
    img_w, img_h = img.size
    t_width, t_height = target_size
    
    # Calculate required scale to fit target size
    subject_width = bbox[2] - bbox[0]
    subject_height = bbox[3] - bbox[1]
    
    # Apply zoom factor
    zoomed_width = subject_width * zoom
    zoomed_height = subject_height * zoom
    
    # Calculate scale to fit target dimensions
    scale = min(t_width/zoomed_width, t_height/zoomed_height)
    
    # Calculate final crop dimensions
    crop_width = int(t_width / scale)
    crop_height = int(t_height / scale)
    
    # Center coordinates with boundary checks
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    
    left = max(0, cx - crop_width//2)
    right = min(img_w, cx + crop_width//2)
    top = max(0, cy - crop_height//2)
    bottom = min(img_h, cy + crop_height//2)
    
    # Adjust if out of bounds
    if right - left < crop_width:
        if left == 0: right = min(crop_width, img_w)
        else: left = max(0, right - crop_width)
    if bottom - top < crop_height:
        if top == 0: bottom = min(crop_height, img_h)
        else: top = max(0, bottom - crop_height)
    
    # Crop and resize
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize(target_size, Image.LANCZOS)

def compress_image(img: Image.Image, max_kb: int):
    rgb_img = img.convert('RGB')
    buffer = io.BytesIO()
    quality = 95
    while quality >= 20:
        buffer.seek(0)
        buffer.truncate()
        quality -= 5
    buffer.seek(0)
    return buffer.getvalue(), size_kb, quality

# Streamlit application
def main():
    # Initialize session state variables
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

    st.set_page_config(
        page_title="Smart Image Resizer",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Global CSS for dark theme
    st.markdown(
    """
    <style>
    /* Dark background for main and sidebar */
    .reportview-container, .main, .css-1lcbmhc {
        background-color: #121212 !important;
        color: #ECECEC !important;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e !important;
    }
    /* Accent color for headings and buttons */
    h1, .st-bx {
        color: #00BFFF !important;
        font-family: 'Courier New', Courier, monospace;
    }
    .stButton>button {
        background-color: #00BFFF !important;
        color: #121212 !important;
        border-radius: 8px;
    }
    /* Style inputs and sliders */
    .stNumberInput > div > div > input {
        background-color: #252525 !important;
        color: #ECECEC !important;
        border: 1px solid #00BFFF;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True
    )


    st.title("Automated Smart Crop & Resize Around Person")
    st.info("💡 Toggle ▸ sidebar for settings: dimensions, margin, file size.")
    st.subheader("Start by uploading images to process.")

    # Sidebar settings & history
    with st.sidebar:
        st.header("⚙️ Settings")
        width = st.number_input("Output Width (px)", 100, 5000, 1200)
        height = st.number_input("Output Height (px)", 100, 5000, 1800)
        max_kb = st.number_input("Max File Size (KB)", 10, 10000, 800)
        margin = st.slider("Crop Margin (px)", 0, 200, 20)
        st.markdown("---")
        if st.button("🧼 Clear History"):
            st.session_state.history.clear()
            st.session_state.uploaded_files = []
            st.session_state.processed_data = None
        st.subheader("Upload History")
        for i, name in enumerate(reversed(st.session_state.history), 1):
            st.text(f"{i}. {name}")
        st.markdown("_Upload history persists until cleared._")

    # File uploader
    uploaded = st.file_uploader("Upload PNG/JPEG Images", type=["png","jpg","jpeg"], accept_multiple_files=True)
    if uploaded and uploaded != st.session_state.uploaded_files:
        st.session_state.uploaded_files = uploaded
        # Update history
        new_names = [f.name for f in uploaded if f.name not in st.session_state.history]
        st.session_state.history.extend(new_names)
        st.session_state.processed_data = None
    files = st.session_state.uploaded_files

    # Collapsible upload list
    if files:
        with st.expander("Uploads", expanded=False):
            for f in files:
                st.write(f.name)

    # Preview gallery (same as Processed Images)
    if files:
        st.subheader("Preview")
        cols = st.columns(3)
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 3]:
                st.image(Image.open(file), caption=file.name, use_container_width=True)

    # Process button & logic
    if files and st.button("🚀 Process All Images"):
        processed = []
        progress = st.progress(0)
        for i, file in enumerate(files):
            img = Image.open(file).convert('RGB')
            cropped = crop_around_subject(img, (width, height), margin)
            resized = resize_with_crop(cropped, (width, height))
            out_bytes, out_kb, q = compress_image(resized, max_kb)
            processed.append((file.name, resized, out_bytes, out_kb, q))
            progress.progress((i+1)/len(files))
        st.session_state.processed_data = processed

    # Processed gallery + download
    data = st.session_state.processed_data
    if data:
        st.subheader("Processed Images")
        cols = st.columns(3)
        for idx, (name, img, _, out_kb, q) in enumerate(data):
            cols[idx % 3].image(img, caption=f"{name} | {out_kb:.1f}KB | Q={q}", use_container_width=True)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for name, _, d, _, _ in data:
                zf.writestr(f"processed_{name.split('.')[0]}.jpg", d)
        buf.seek(0)
        st.download_button("📥 Download ZIP", data=buf, file_name="processed_images.zip", mime="application/zip")

if __name__ == "__main__":
    main()
