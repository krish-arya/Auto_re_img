import streamlit as st
from PIL import Image, ImageOps, ImageColor
import io, zipfile, cv2, numpy as np
from rembg import remove
from ultralytics import YOLO
from typing import Tuple, Optional

# ========== Configuration ==========
st.set_page_config(
    page_title="AI Photo Editor Pro",
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# ========== Model Loading ==========
@st.cache_resource
def load_yolo_model():
    try:
        return YOLO('yolov8n-seg.pt')  # Use segmentation model
    except Exception as e:
        st.error(f"Failed to load YOLO model: {str(e)}")
        return None

# ========== Core Functions ==========
def enhanced_subject_detection(model, img: Image.Image) -> Optional[Tuple[int]]:
    """Combine YOLO detection with rembg for accurate subject masking"""
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

    # Fallback to rembg with edge refinement
    bg_removed = remove(img, post_process_mask=True)
    alpha = bg_removed.split()[-1]
    bbox = alpha.getbbox()
    
    if bbox:
        # Expand bbox by 5% for safety
        w, h = img.size
        dx = int((bbox[2] - bbox[0]) * 0.05)
        dy = int((bbox[3] - bbox[1]) * 0.05)
        return (
            max(0, bbox[0] - dx),
            max(0, bbox[1] - dy),
            min(w, bbox[2] + dx),
            min(h, bbox[3] + dy)
        )
    return None

def smart_crop_with_zoom(img: Image.Image, bbox: Tuple[int], zoom_factor: float):
    """Precision cropping with boundary checks"""
    img_w, img_h = img.size
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    
    # Calculate zoomed dimensions
    base_width = bbox[2] - bbox[0]
    base_height = bbox[3] - bbox[1]
    zoom_w = int(base_width * zoom_factor)
    zoom_h = int(base_height * zoom_factor)
    
    # Maintain aspect ratio of original detection
    aspect_ratio = base_width / base_height
    if zoom_w / aspect_ratio > zoom_h:
        zoom_h = int(zoom_w / aspect_ratio)
    else:
        zoom_w = int(zoom_h * aspect_ratio)
    
    # Calculate safe crop area
    left = max(0, cx - zoom_w//2)
    right = min(img_w, cx + zoom_w//2)
    top = max(0, cy - zoom_h//2)
    bottom = min(img_h, cy + zoom_h//2)
    
    # Adjust if we hit image boundaries
    if right - left < zoom_w:
        if left == 0: right = min(zoom_w, img_w)
        else: left = max(0, right - zoom_w)
    if bottom - top < zoom_h:
        if top == 0: bottom = min(zoom_h, img_h)
        else: top = max(0, bottom - zoom_h)
    
    return img.crop((left, top, right, bottom))

def intelligent_resize(img: Image.Image, target_size: Tuple[int], bg_color: Tuple[int], use_inpainting: bool):
    """Resize with context-aware padding"""
    img = img.convert("RGB")
    
    # Calculate scaling while maintaining aspect ratio
    ratio = min(target_size[0]/img.width, target_size[1]/img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    resized = img.resize(new_size, Image.LANCZOS)
    
    if use_inpainting:
        # Convert to OpenCV format for inpainting
        cv_img = cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)
        target_shape = (target_size[1], target_size[0], 3)
        padded = np.full(target_shape, bg_color, dtype=np.uint8)
        
        # Center the resized image
        y_offset = (target_size[1] - new_size[1]) // 2
        x_offset = (target_size[0] - new_size[0]) // 2
        padded[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = cv_img
        
        # Create inpainting mask
        mask = np.zeros((target_shape[0], target_shape[1]), dtype=np.uint8)
        mask[:y_offset, :] = 255
        mask[y_offset+new_size[1]:, :] = 255
        mask[:, :x_offset] = 255
        mask[:, x_offset+new_size[0]:] = 255
        
        # Inpainting with edge preservation
        inpainted = cv2.inpaint(padded, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    
    # Standard padding
    return ImageOps.pad(resized, target_size, color=bg_color, centering=(0.5, 0.5))

def optimize_image(img: Image.Image, max_size_kb: int) -> io.BytesIO:
    """Adaptive compression with quality tuning"""
    buffer = io.BytesIO()
    quality = 95
    img.save(buffer, "JPEG", quality=quality, optimize=True, progressive=True)
    
    while buffer.tell()/1024 > max_size_kb and quality > 10:
        buffer.seek(0)
        buffer.truncate()
        quality -= 5
        img.save(buffer, "JPEG", quality=quality, optimize=True, progressive=True)
    
    buffer.seek(0)
    return buffer

# ========== Streamlit UI ==========
model = load_yolo_model()

with st.sidebar:
    st.header("⚙️ Settings")
    target_width = st.number_input("Output Width", 512, 4096, 1200)
    target_height = st.number_input("Output Height", 512, 4096, 1800)
    zoom_factor = st.slider("Zoom Level", 1.0, 3.0, 1.2, 0.1)
    max_file_size = st.number_input("Max File Size (KB)", 50, 10240, 500)
    bg_color = st.color_picker("Background Color", "#FFFFFF")
    use_inpainting = st.checkbox("AI Background Extension", True)
    process_all = st.checkbox("Process Multiple Files", True)

st.title("📸 AI-Powered Photo Editor")
uploaded_files = st.file_uploader("Upload Images", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 Start Processing" if process_all else "✨ Process Image"):
        processed = []
        progress_bar = st.progress(0)
        
        for idx, file in enumerate(uploaded_files):
            try:
                # Load and process image
                img = Image.open(file).convert("RGB")
                bbox = enhanced_subject_detection(model, img)
                
                if not bbox:
                    st.warning(f"No subject detected in {file.name}, using center crop")
                    bbox = (img.width//4, img.height//4, 3*img.width//4, 3*img.height//4)
                
                cropped = smart_crop_with_zoom(img, bbox, zoom_factor)
                final_img = intelligent_resize(
                    cropped,
                    (target_width, target_height),
                    ImageColor.getrgb(bg_color),
                    use_inpainting
                )
                
                # Compress and save
                buffer = optimize_image(final_img, max_file_size)
                processed.append((file.name, final_img, buffer))
                progress_bar.progress((idx+1)/len(uploaded_files))
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
        
        st.session_state.processed_images = processed
        st.success(f"Processed {len(processed)} images successfully!")

if "processed_images" in st.session_state:
    st.header("🎨 Processed Results")
    cols = st.columns(3)
    
    for idx, (name, img, buffer) in enumerate(st.session_state.processed_images):
        with cols[idx % 3]:
            st.image(img, caption=name, use_column_width=True)
            st.download_button(
                f"Download {name.split('.')[0]}",
                data=buffer.getvalue(),
                file_name=f"enhanced_{name}",
                mime="image/jpeg",
                key=f"dl_{idx}"
            )
    
    # Create ZIP archive for all images
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for name, _, buffer in st.session_state.processed_images:
            zip_file.writestr(f"enhanced_{name}", buffer.getvalue())
    
    st.download_button(
        "📦 Download All as ZIP",
        data=zip_buffer.getvalue(),
        file_name="enhanced_images.zip",
        mime="application/zip"
    )