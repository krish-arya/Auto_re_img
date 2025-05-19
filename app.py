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
    initial_sidebar_state="expanded"
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
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if model:
        results = model.predict(img_cv, classes=0, verbose=False)
        for r in results:
            if r.masks is not None:
                masks = r.masks.xy
                if len(masks) > 0:
                    largest_mask = max(masks, key=lambda m: cv2.contourArea(m))
                    x, y, w, h = cv2.boundingRect(largest_mask.astype(np.int32))
                    return (x, y, x + w, y + h)
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
    img_w, img_h = img.size
    t_width, t_height = target_size
    subject_width = bbox[2] - bbox[0]
    subject_height = bbox[3] - bbox[1]
    zoomed_width = subject_width * zoom
    zoomed_height = subject_height * zoom
    scale = min(t_width / zoomed_width, t_height / zoomed_height)
    crop_width = int(t_width / scale)
    crop_height = int(t_height / scale)
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    left = max(0, cx - crop_width // 2)
    right = min(img_w, cx + crop_width // 2)
    top = max(0, cy - crop_height // 2)
    bottom = min(img_h, cy + crop_height // 2)
    if right - left < crop_width:
        if left == 0: right = min(crop_width, img_w)
        else: left = max(0, right - crop_width)
    if bottom - top < crop_height:
        if top == 0: bottom = min(crop_height, img_h)
        else: top = max(0, bottom - crop_height)
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize(target_size, Image.LANCZOS)

def optimize_image(img: Image.Image, max_size_kb: int) -> io.BytesIO:
    buffer = io.BytesIO()
    quality = 95
    img.save(buffer, "JPEG", quality=quality, optimize=True, progressive=True)
    while buffer.tell() / 1024 > max_size_kb and quality > 10:
        buffer.seek(0)
        buffer.truncate()
        quality -= 5
        img.save(buffer, "JPEG", quality=quality, optimize=True, progressive=True)
    buffer.seek(0)
    return buffer

# ========== UI and App State ==========
model = load_yolo_model()

# Track uploader widget state
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

with st.sidebar:
    st.header("⚙️ Settings")
    target_width = st.number_input("Output Width", 512, 4096, 1200)
    target_height = st.number_input("Output Height", 512, 4096, 1800)
    zoom_factor = st.slider("Zoom Level", 0.5, 3.0, 1.2, 0.1)
    max_file_size = st.number_input("Max File Size (KB)", 50, 10240, 500)
    process_all = st.checkbox("Process Multiple Files", True)

    if st.button("🗑️ Clear Uploaded Files"):
        st.session_state.upload_key += 1
        st.session_state.pop("processed_images", None)
        st.rerun()

st.title("📸 AI-Powered Photo Cropper")

uploaded_files = st.file_uploader(
    "Upload Images", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.upload_key}"
)

# ========== Preview Uploaded Files ==========
if uploaded_files:
    with st.expander("🔍 Preview Uploaded Images", expanded=True):
        cols = st.columns(3)
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 3]:
                st.image(Image.open(file), caption=file.name, use_container_width=True)

# ========== Process Images ==========
if uploaded_files and st.button("🚀 Start Processing" if process_all else "✨ Process Image"):
    processed = []
    progress_bar = st.progress(0)
    
    for idx, file in enumerate(uploaded_files):
        try:
            img = Image.open(file).convert("RGB")
            bbox = enhanced_subject_detection(model, img)
            if not bbox:
                st.warning(f"No subject detected in {file.name}, using center crop")
                bbox = (img.width // 4, img.height // 4, 3 * img.width // 4, 3 * img.height // 4)
            final_img = smart_crop_to_target(img, bbox, (target_width, target_height), zoom_factor)
            buffer = optimize_image(final_img, max_file_size)
            processed.append((file.name, final_img, buffer))
            progress_bar.progress((idx + 1) / len(uploaded_files))
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")

    st.session_state.processed_images = processed
    st.success(f"Processed {len(processed)} images successfully!")

# ========== Show Processed Results ==========
if "processed_images" in st.session_state:
    st.header("🎨 Processed Results")
    cols = st.columns(3)
    for idx, (name, img, buffer) in enumerate(st.session_state.processed_images):
        with cols[idx % 3]:
            st.image(img, caption=name, use_container_width=True)
            st.download_button(
                f"Download {name.split('.')[0]}",
                data=buffer.getvalue(),
                file_name=f"cropped_{name}",
                mime="image/jpeg",
                key=f"dl_{idx}"
            )
    # ZIP download
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for name, _, buffer in st.session_state.processed_images:
            zip_file.writestr(f"cropped_{name}", buffer.getvalue())
    st.download_button(
        "📦 Download All as ZIP",
        data=zip_buffer.getvalue(),
        file_name="cropped_images.zip",
        mime="application/zip"
    )
