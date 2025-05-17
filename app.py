import streamlit as st
from PIL import Image
from rembg import remove
import io
import zipfile
import base64

# Detect and smart-crop around the person, adjusting box to target aspect ratio
# (Logic unchanged)
def crop_around_subject(img: Image.Image, target_size: tuple, margin: int = 20):
    input_buffer = io.BytesIO()
    img.save(input_buffer, format="PNG")
    fg_bytes = remove(input_buffer.getvalue())
    fg_img = Image.open(io.BytesIO(fg_bytes)).convert("RGBA")
    alpha = fg_img.split()[-1]
    bbox = alpha.getbbox() or (0, 0, img.width, img.height)
    left, upper, right, lower = bbox
    left, upper = max(left - margin, 0), max(upper - margin, 0)
    right, lower = min(right + margin, img.width), min(lower + margin, img.height)
    target_w, target_h = target_size
    target_ratio = target_w / target_h
    box_w, box_h = right - left, lower - upper
    box_ratio = box_w / box_h
    if box_ratio > target_ratio:
        new_h = int(box_w / target_ratio)
        expand = (new_h - box_h) // 2
        upper, lower = max(upper - expand, 0), min(lower + expand, img.height)
    else:
        new_w = int(box_h * target_ratio)
        expand = (new_w - box_w) // 2
        left, right = max(left - expand, 0), min(right + expand, img.width)
    return img.crop((left, upper, right, lower))

def resize_with_crop(img: Image.Image, target_size: tuple):
    target_ratio = target_size[0] / target_size[1]
    img_ratio = img.width / img.height
    if img_ratio > target_ratio:
        new_width = int(img.height * target_ratio)
        offset = (img.width - new_width) // 2
        img = img.crop((offset, 0, offset + new_width, img.height))
    else:
        new_height = int(img.width / target_ratio)
        offset = (img.height - new_height) // 2
        img = img.crop((0, offset, img.width, offset + new_height))
    return img.resize(target_size, Image.LANCZOS)

def compress_image(img: Image.Image, max_kb: int):
    rgb_img = img.convert('RGB')
    buffer = io.BytesIO()
    quality = 95
    while quality >= 20:
        buffer.seek(0)
        buffer.truncate()
        rgb_img.save(buffer, format='JPEG', quality=quality, optimize=True, progressive=True)
        size_kb = buffer.tell() / 1024
        if size_kb <= max_kb:
            buffer.seek(0)
            return buffer.getvalue(), size_kb, quality
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
        for idx, file in enumerate(files):
            img = Image.open(file).convert('RGB')
            cols[idx % 3].image(img, caption=file.name, use_container_width=True)

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
