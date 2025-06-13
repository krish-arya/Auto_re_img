import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"

import streamlit as st

from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageEnhance
import io, zipfile
import cv2
import numpy as np
from rembg import remove
from ultralytics import YOLO
from typing import Tuple, Optional, Union, List
import ssl
import imageio
from imageio_ffmpeg import get_ffmpeg_exe
import subprocess
import warnings
import tempfile
import math
warnings.filterwarnings("ignore")

ssl._create_default_https_context = ssl._create_unverified_context

# ========== Page Configuration ==========
st.set_page_config(
    page_title="AI Cropper + Brand Generator & Slideshow",
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="collapsed"
)

# ========== Model Loading ========== 
@st.cache_resource
def load_yolo_model():
    try:
        return YOLO("yolov8n-seg.pt")
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

model = load_yolo_model()

# ========== Enhanced Transition Functions ==========

def apply_fade_transition(img1: np.ndarray, img2: np.ndarray, progress: float) -> np.ndarray:
    """Simple fade transition between two images"""
    return cv2.addWeighted(img1, 1 - progress, img2, progress, 0)

def apply_slide_transition(img1: np.ndarray, img2: np.ndarray, progress: float, direction: str = 'left') -> np.ndarray:
    """Slide transition in specified direction"""
    h, w = img1.shape[:2]
    result = np.zeros_like(img1)
    
    if direction == 'left':
        offset = int(w * progress)
        if offset < w:
            result[:, :w-offset] = img1[:, offset:]
            result[:, w-offset:] = img2[:, :offset]
    elif direction == 'right':
        offset = int(w * progress)
        if offset < w:
            result[:, offset:] = img1[:, :w-offset]
            result[:, :offset] = img2[:, w-offset:]
    elif direction == 'up':
        offset = int(h * progress)
        if offset < h:
            result[:h-offset, :] = img1[offset:, :]
            result[h-offset:, :] = img2[:offset, :]
    elif direction == 'down':
        offset = int(h * progress)
        if offset < h:
            result[offset:, :] = img1[:h-offset, :]
            result[:offset, :] = img2[h-offset:, :]
    
    return result

def apply_zoom_transition(img1: np.ndarray, img2: np.ndarray, progress: float) -> np.ndarray:
    """Zoom out from img1 while fading in img2"""
    h, w = img1.shape[:2]
    
    # Zoom effect on img1
    scale = 1 + progress * 0.5
    center_x, center_y = w // 2, h // 2
    
    # Create transformation matrix
    M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
    zoomed_img1 = cv2.warpAffine(img1, M, (w, h))
    
    # Fade transition
    return apply_fade_transition(zoomed_img1, img2, progress)

def apply_rotate_transition(img1: np.ndarray, img2: np.ndarray, progress: float) -> np.ndarray:
    """Rotate img1 while fading to img2"""
    h, w = img1.shape[:2]
    angle = progress * 180
    
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img1 = cv2.warpAffine(img1, M, (w, h))
    
    return apply_fade_transition(rotated_img1, img2, progress)

def apply_blur_transition(img1: np.ndarray, img2: np.ndarray, progress: float) -> np.ndarray:
    """Blur img1 while transitioning to img2"""
    blur_strength = int(progress * 20) * 2 + 1  # Ensure odd number
    if blur_strength > 1:
        blurred_img1 = cv2.GaussianBlur(img1, (blur_strength, blur_strength), 0)
    else:
        blurred_img1 = img1
    
    return apply_fade_transition(blurred_img1, img2, progress)

def apply_wipe_transition(img1: np.ndarray, img2: np.ndarray, progress: float, direction: str = 'horizontal') -> np.ndarray:
    """Wipe transition with smooth edge"""
    h, w = img1.shape[:2]
    result = img1.copy()
    
    if direction == 'horizontal':
        wipe_pos = int(w * progress)
        if wipe_pos > 0:
            # Create smooth edge with gradient
            edge_width = min(20, wipe_pos)
            for i in range(edge_width):
                alpha = i / edge_width
                if wipe_pos - edge_width + i < w:
                    result[:, wipe_pos - edge_width + i] = cv2.addWeighted(
                        img1[:, wipe_pos - edge_width + i], 1 - alpha,
                        img2[:, wipe_pos - edge_width + i], alpha, 0
                    )
            if wipe_pos < w:
                result[:, :wipe_pos - edge_width] = img2[:, :wipe_pos - edge_width]
    else:  # vertical
        wipe_pos = int(h * progress)
        if wipe_pos > 0:
            edge_width = min(20, wipe_pos)
            for i in range(edge_width):
                alpha = i / edge_width
                if wipe_pos - edge_width + i < h:
                    result[wipe_pos - edge_width + i, :] = cv2.addWeighted(
                        img1[wipe_pos - edge_width + i, :], 1 - alpha,
                        img2[wipe_pos - edge_width + i, :], alpha, 0
                    )
            if wipe_pos < h:
                result[:wipe_pos - edge_width, :] = img2[:wipe_pos - edge_width, :]
    
    return result

def apply_circle_transition(img1: np.ndarray, img2: np.ndarray, progress: float) -> np.ndarray:
    """Circular reveal transition"""
    h, w = img1.shape[:2]
    center_x, center_y = w // 2, h // 2
    max_radius = int(math.sqrt(center_x**2 + center_y**2))
    current_radius = int(max_radius * progress)
    
    # Create circular mask
    y, x = np.ogrid[:h, :w]
    mask = (x - center_x)**2 + (y - center_y)**2 <= current_radius**2
    
    result = img1.copy()
    result[mask] = img2[mask]
    
    # Smooth edge
    edge_thickness = 10
    edge_mask = ((x - center_x)**2 + (y - center_y)**2 <= (current_radius + edge_thickness)**2) & \
                ((x - center_x)**2 + (y - center_y)**2 > (current_radius - edge_thickness)**2)
    
    if np.any(edge_mask):
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        alpha = np.clip((distances - current_radius + edge_thickness) / (2 * edge_thickness), 0, 1)
        alpha = alpha[edge_mask]
        
        result[edge_mask] = (img1[edge_mask] * alpha[:, np.newaxis] + 
                           img2[edge_mask] * (1 - alpha[:, np.newaxis])).astype(np.uint8)
    
    return result

def apply_pixelate_transition(img1: np.ndarray, img2: np.ndarray, progress: float) -> np.ndarray:
    """Pixelate img1 while transitioning to img2"""
    h, w = img1.shape[:2]
    
    # Pixelate img1
    pixel_size = max(1, int(progress * 20))
    small_img1 = cv2.resize(img1, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated_img1 = cv2.resize(small_img1, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return apply_fade_transition(pixelated_img1, img2, progress)

# ========== Utility Functions ==========

def compute_center_of_bbox(bbox: Tuple[int]) -> Tuple[int, int]:
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

def enhanced_subject_detection(model, img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    First try YOLOv8 segmentation; if that fails, fall back to rembg mask bounding box.
    Returns (x0, y0, x1, y1) or None.
    """
    if model is None:
        return None
        
    try:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = model.predict(img_cv, classes=0, verbose=False)
        for r in results:
            if r.masks is not None:
                masks = r.masks.xy
                if len(masks) > 0:
                    largest_mask = max(masks, key=lambda m: cv2.contourArea(m))
                    x, y, w, h = cv2.boundingRect(largest_mask.astype(np.int32))
                    return (x, y, x + w, y + h)
    except Exception as e:
        st.warning(f"YOLO detection failed: {e}")

    # Fallback: remove background and get alpha mask bbox
    try:
        bg_removed = remove(img, post_process_mask=True)
        alpha = bg_removed.split()[-1]
        bbox = alpha.getbbox()
        if bbox:
            dx = int((bbox[2] - bbox[0]) * 0.05)
            dy = int((bbox[3] - bbox[1]) * 0.05)
            x0 = max(0, bbox[0] - dx)
            y0 = max(0, bbox[1] - dy)
            x1 = min(img.width, bbox[2] + dx)
            y1 = min(img.height, bbox[3] + dy)
            return (x0, y0, x1, y1)
    except Exception as e:
        st.warning(f"Background removal failed: {e}")

    return None

def smart_crop_with_headspace(
    img: Image.Image,
    bbox: Tuple[int, int, int, int],
    target_size: Tuple[int, int],
    zoom: float,
    headspace_top: Union[int, float],
    headspace_bottom: Union[int, float],
    use_percent: bool = False
) -> Image.Image:
    """
    Crop around the subject bounding box with optional headspace above/below,
    then resize exactly to target_size.
    """
    img_w, img_h = img.size
    t_w, t_h = target_size

    subject_w = bbox[2] - bbox[0]
    subject_h = bbox[3] - bbox[1]

    zoomed_w = subject_w * zoom
    zoomed_h = subject_h * zoom

    scale = min(t_w / zoomed_w, t_h / zoomed_h)
    crop_w = int(t_w / scale)
    crop_h = int(t_h / scale)

    top_extra = int((headspace_top * subject_h) / 100) if use_percent else int(headspace_top)
    bottom_extra = int((headspace_bottom * subject_h) / 100) if use_percent else int(headspace_bottom)

    cx, cy = compute_center_of_bbox(bbox)
    top = max(0, cy - crop_h // 2 - top_extra)
    bottom = min(img_h, cy + crop_h // 2 + bottom_extra)
    left = max(0, cx - crop_w // 2)
    right = min(img_w, cx + crop_w // 2)

    # Adjust if near edges
    if right - left < crop_w:
        if left == 0:
            right = min(crop_w, img_w)
        else:
            left = max(0, right - crop_w)
    if bottom - top < crop_h:
        if top == 0:
            bottom = min(crop_h, img_h)
        else:
            top = max(0, bottom - crop_h)

    cropped = img.crop((left, top, right, bottom))
    return cropped.resize(target_size, Image.LANCZOS)

def optimize_image(img: Image.Image, max_size_kb: int) -> io.BytesIO:
    """
    Iteratively reduce JPEG quality until under max_size_kb. Returns a BytesIO.
    """
    buffer = io.BytesIO()
    quality = 95
    try:
        img.save(buffer, "JPEG", quality=quality, optimize=True, progressive=True)
        while (buffer.tell() / 1024) > max_size_kb and quality > 10:
            buffer.seek(0)
            buffer.truncate()
            quality -= 5
            img.save(buffer, "JPEG", quality=quality, optimize=True, progressive=True)
    except Exception as e:
        st.error(f"Failed to optimize image: {e}")
        img.save(buffer, "JPEG", quality=85)
    
    buffer.seek(0)
    return buffer

def apply_branding(
    img: Image.Image,
    logo: Optional[Image.Image],
    **kwargs
) -> Image.Image:
    """
    Apply padding, logo, and optional text overlay.
    """
    try:
        composite = img.convert("RGBA")

        # 1) Padding
        if kwargs.get("add_padding", False):
            pad = kwargs.get("padding", 0)
            new_w = composite.width + 2 * pad
            new_h = composite.height + 2 * pad
            color = kwargs.get("padding_color", (255, 255, 255, 0))
            base = Image.new("RGBA", (new_w, new_h), color)
            base.paste(composite, (pad, pad))
            composite = base

        # 2) Logo
        if logo is not None:
            logo = logo.convert("RGBA")
            logo_w = int((kwargs.get("logo_scale", 25) / 100) * composite.width)
            logo_h = int(logo_w * (logo.height / logo.width))
            logo_resized = logo.resize((logo_w, logo_h), Image.LANCZOS)
            x_px = int((kwargs.get("x_offset", 50) / 100) * (composite.width - logo_w))
            y_px = int((kwargs.get("y_offset", 90) / 100) * (composite.height - logo_h))
            composite.paste(logo_resized, (x_px, y_px), logo_resized)

        # 3) Text
        if kwargs.get("add_text", False) and kwargs.get("text", ""):
            draw = ImageDraw.Draw(composite)
            try:
                font = ImageFont.truetype("arial.ttf", kwargs.get("font_size", 40))
            except:
                font = ImageFont.load_default()
            tx = int((kwargs.get("text_x", 5) / 100) * composite.width)
            ty = int((kwargs.get("text_y", 5) / 100) * composite.height)
            draw.text((tx, ty), kwargs.get("text", ""), fill=kwargs.get("text_color", "#000000"), font=font)

        return composite.convert("RGB")
    except Exception as e:
        st.error(f"Failed to apply branding: {e}")
        return img

def preprocess_uploaded_image(img: Image.Image, max_dim: int = 2048) -> Image.Image:
    """
    Resize any side to max_dim if needed, to keep memory usage sane.
    """
    try:
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        return img.convert("RGB")
    except Exception as e:
        st.error(f"Failed to preprocess image: {e}")
        return img

def safe_image_resize_and_crop(img: Image.Image, target_w: int, target_h: int) -> np.ndarray:
    """Safely resize and crop image to target dimensions"""
    try:
        # Calculate aspect ratios
        img_ratio = img.width / img.height
        target_ratio = target_w / target_h
        
        # Resize to fill target dimensions
        if img_ratio > target_ratio:
            # Image is wider than target
            new_h = target_h
            new_w = int(img_ratio * target_h)
        else:
            # Image is taller than target
            new_w = target_w
            new_h = int(target_w / img_ratio)
        
        # Resize image
        resized_img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Center crop to exact target size
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        cropped_img = resized_img.crop((left, top, left + target_w, top + target_h))
        
        return np.array(cropped_img)
    except Exception as e:
        st.error(f"Failed to resize/crop image: {e}")
        # Return a black frame as fallback
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

# ========== Session State & Mode Selection ==========

# 1) Keep a dynamic-key so we can CLEAR the uploader when needed:
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

# 2) Sidebar: choose mode and Clear button
with st.sidebar:
    st.markdown("## 🎛️ Select App Mode")
    mode = st.selectbox(
        "Choose an action:",
        ["🎯 Smart Cropper + Branding", "🎞️ Instagram Slideshow"],
        index=0,
    )
    st.markdown("---")
    if st.button("🗑️ Clear Uploaded Files"):
        st.session_state.upload_key += 1
        st.rerun()

# ========== Common Upload Section (shared by both modes) ==========

st.title("📸 AI‑Powered Smart Cropper + Slideshow Generator")
st.info("Use the sidebar to pick a mode and upload images.", icon="🛠️")

uploaded_files = st.file_uploader(
    "📸 Upload Image(s) (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.upload_key}"
)

# Convert UploadedFile to PIL so we can preview quickly
def load_image_from_uploaded(upl) -> Image.Image:
    try:
        return Image.open(upl).convert("RGB")
    except Exception as e:
        st.error(f"Failed to load image {upl.name}: {e}")
        return None

if uploaded_files:
    st.subheader("🔍 Uploaded Image Preview")
    valid_files = []
    cols = st.columns(min(4, len(uploaded_files)))
    for idx, upl in enumerate(uploaded_files):
        img = load_image_from_uploaded(upl)
        if img is not None:
            img = preprocess_uploaded_image(img)
            cols[idx % len(cols)].image(img, use_container_width=True, caption=upl.name)
            valid_files.append(upl)
    
    # Update uploaded_files to only include valid files
    uploaded_files = valid_files

# ========== Mode 1: Smart Cropper + Brand Generator ==========

if mode == "🎯 Smart Cropper + Branding":
    st.sidebar.markdown("## ✂️ Smart Crop Settings")
    with st.sidebar.expander("📐 Output Dimensions"):
        target_width = st.number_input("Width", 512, 4096, 1200, step=100)
        target_height = st.number_input("Height", 512, 4096, 1800, step=100)
        zoom_factor = st.slider("Zoom Level", 0.5, 3.0, 1.2, 0.1)
        st.markdown("---")
        max_size_kb = st.number_input("Max File Size (KB)", 100, 5000, 800, step=50)

    with st.sidebar.expander("🧠 Headspace & Cropping"):
        use_percent = st.checkbox("Use Percent for Headspace")
        top_space = st.number_input("Top Headspace", 0, 1000, 10)
        bottom_space = st.number_input("Bottom Headspace", 0, 1000, 10)

    st.sidebar.markdown("## 🎨 Branding Options")
    with st.sidebar.expander("🏷️ Logo Settings"):
        logo_file = st.file_uploader("Upload Logo (PNG)", type=["png"])
        logo_scale = st.slider("Logo Size (% of width)", 5, 50, 25)
        x_offset = st.slider("Logo Horizontal Pos (%)", 0, 100, 50)
        y_offset = st.slider("Logo Vertical Pos (%)", 0, 100, 90)

    with st.sidebar.expander("🔤 Text Overlay"):
        add_text = st.checkbox("Add Text")
        if add_text:
            text = st.text_input("Text Content", "Your Brand Message")
            font_size = st.slider("Font Size", 10, 150, 90)
            text_color = st.color_picker("Text Color", "#000000")
            text_x = st.slider("Text Horizontal Pos (%)", 0, 100, 50)
            text_y = st.slider("Text Vertical Pos (%)", 0, 100, 90)
        else:
            text = ""
            font_size = 40
            text_color = "#000000"
            text_x = 5
            text_y = 5

    with st.sidebar.expander("🧱 Padding"):
        add_padding = st.checkbox("Add Padding")
        if add_padding:
            padding = st.slider("Padding (px)", 0, 300, 50)
            padding_color = st.color_picker("Padding Color", "#FFFFFF")
        else:
            padding = 0
            padding_color = "#FFFFFF"
            add_padding = False

    # Only show "Process" if there is at least one upload
    if uploaded_files:
        if st.button("🚀 Process Images"):
            results = []
            logo_img = None
            if logo_file:
                try:
                    logo_img = Image.open(logo_file).convert("RGBA")
                except Exception as e:
                    st.error(f"Failed to load logo: {e}")

            progress = st.progress(0, text="Processing…")
            
            for i, upl in enumerate(uploaded_files):
                try:
                    base_img = preprocess_uploaded_image(load_image_from_uploaded(upl))
                    if base_img is None:
                        continue
                        
                    if max(base_img.size) > 3000:
                        base_img = base_img.resize((base_img.width // 2, base_img.height // 2), Image.LANCZOS)

                    bbox = enhanced_subject_detection(model, base_img)
                    if not bbox:
                        # fallback to center if detection fails
                        w, h = base_img.size
                        bbox = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)

                    cropped = smart_crop_with_headspace(
                        base_img, bbox, (target_width, target_height),
                        zoom_factor, top_space, bottom_space, use_percent
                    )

                    branded_img = apply_branding(
                        cropped, logo_img,
                        logo_scale=logo_scale, x_offset=x_offset, y_offset=y_offset,
                        add_text=add_text, text=text, font_size=font_size,
                        text_color=text_color, text_x=text_x, text_y=text_y,
                        add_padding=add_padding, padding=padding, padding_color=padding_color
                    )

                    buf = optimize_image(branded_img, max_size_kb)
                    results.append((upl.name, branded_img, buf))
                    
                except Exception as e:
                    st.error(f"Failed to process {upl.name}: {e}")
                    continue
                    
                progress.progress((i + 1) / len(uploaded_files), text=f"Processed {i+1}/{len(uploaded_files)}")

            progress.empty()

            if results:
                st.subheader("🎨 Branded Output Preview")
                preview_cols = st.columns(min(4, len(results)))
                for idx, (fname, img_obj, buff) in enumerate(results):
                    with preview_cols[idx % len(preview_cols)]:
                        st.image(img_obj, caption=fname, use_container_width=True)
                        st.download_button(
                            label="⬇️ Download",
                            data=buff.getvalue(),
                            file_name=f"branded_{fname}",
                            mime="image/jpeg",
                            key=f"download_{idx}"
                        )

                # ZIP all
                zip_buf = io.BytesIO()
                try:
                    with zipfile.ZipFile(zip_buf, "w") as zf:
                        for fname, _, buff in results:
                            zf.writestr(f"branded_{fname}", buff.getvalue())
                    zip_buf.seek(0)
                    st.download_button(
                        "📦 Download All as ZIP",
                        data=zip_buf.getvalue(),
                        file_name="branded_images.zip",
                        mime="application/zip"
                    )
                except Exception as e:
                    st.error(f"Failed to create ZIP file: {e}")
            else:
                st.error("No images were successfully processed.")

    else:
        st.info("Upload at least one image to start cropping + branding.")


# ========== Mode 2: Instagram⁠‑Ready Slideshow Generator ==========

if mode == "🎞️ Instagram Slideshow":
    st.sidebar.markdown("## 🎞️ Slideshow Options")
    
    with st.sidebar.expander("⚙️ Basic Settings"):
        slide_duration = st.slider(
            "Seconds per Slide",
            1, 10, 3,
            help="How many seconds each image should be shown"
        )
        
        resolution = st.selectbox(
            "Output Resolution",
            [
                "1080×1080 (Square)",
                "1080×1350 (Portrait)", 
                "1080×1920 (Vertical)"
            ],
        )
        
        fps = st.selectbox("Frame Rate", [24, 30, 60], index=1)
    
    with st.sidebar.expander("✨ Transition Effects"):
        transition_duration = st.slider(
            "Transition Duration (seconds)",
            0.2, 2.0, 0.8, 0.1,
            help="How long each transition should last"
        )
        
        transition_type = st.selectbox(
            "Transition Type",
            [
                "🌟 Auto Mix (Random)",
                "🌅 Fade",
                "➡️ Slide Left",
                "⬅️ Slide Right", 
                "⬆️ Slide Up",
                "⬇️ Slide Down",
                "🔍 Zoom",
                "🔄 Rotate",
                "💫 Blur",
                "🧹 Wipe Horizontal",
                "🧹 Wipe Vertical",
                "⭕ Circle Reveal",
                "🎮 Pixelate"
            ]
        )
        
        use_ken_burns = st.checkbox(
            "🎬 Ken Burns Effect",
            help="Add subtle zoom/pan motion to static images"
        )

    # Parse resolution
    if resolution.startswith("1080×1080"):
        out_w, out_h = 1080, 1080
    elif resolution.startswith("1080×1350"):
        out_w, out_h = 1080, 1350
    else:  # 1080×1920
        out_w, out_h = 1080, 1920

    # 'uploaded_files' provided elsewhere in the app
    if not uploaded_files:
        st.info("📂 Upload at least one image to generate a slideshow.")
    elif st.button("▶️ Generate & Preview Slideshow"):
        
        if len(uploaded_files) < 2 and transition_type != "🌅 Fade":
            st.warning("⚠️ You need at least 2 images for transitions. Using fade transition for single image.")
        
        # Calculate frame counts
        transition_frames = int(transition_duration * fps) if len(uploaded_files) > 1 else 0
        static_frames = int(slide_duration * fps)
        total_frames = len(uploaded_files) * static_frames + (len(uploaded_files) - 1) * transition_frames
        
        progress_bar = st.progress(0, text="Loading images...")
        
        # Load and process all images first
        processed_images = []
        for i, upl in enumerate(uploaded_files):
            try:
                img = load_image_from_uploaded(upl)
                if img is None:
                    st.warning(f"⚠ Skipping {upl.name} - failed to load")
                    continue
                
                # Process image to target dimensions
                frame = safe_image_resize_and_crop(img, out_w, out_h)
                processed_images.append(frame)
                
                progress_bar.progress((i + 1) / len(uploaded_files) * 0.3, 
                                    text=f"Loading images... {i+1}/{len(uploaded_files)}")
                
            except Exception as e:
                st.warning(f"⚠ Could not process {upl.name}: {e}")
                continue

        if not processed_images:
            st.error("❌ No valid images to make slideshow.")
            st.stop()

        # Generate transition mapping
        transition_functions = {
            "🌅 Fade": apply_fade_transition,
            "➡️ Slide Left": lambda img1, img2, p: apply_slide_transition(img1, img2, p, 'left'),
            "⬅️ Slide Right": lambda img1, img2, p: apply_slide_transition(img1, img2, p, 'right'),
            "⬆️ Slide Up": lambda img1, img2, p: apply_slide_transition(img1, img2, p, 'up'),
            "⬇️ Slide Down": lambda img1, img2, p: apply_slide_transition(img1, img2, p, 'down'),
            "🔍 Zoom": apply_zoom_transition,
            "🔄 Rotate": apply_rotate_transition,
            "💫 Blur": apply_blur_transition,
            "🧹 Wipe Horizontal": lambda img1, img2, p: apply_wipe_transition(img1, img2, p, 'horizontal'),
            "🧹 Wipe Vertical": lambda img1, img2, p: apply_wipe_transition(img1, img2, p, 'vertical'),
            "⭕ Circle Reveal": apply_circle_transition,
            "🎮 Pixelate": apply_pixelate_transition
        }

        # Auto mix transitions
        auto_transitions = list(transition_functions.keys())
        
        progress_bar.progress(0.4, text="Generating frames...")
        
        frames = []
        current_frame = 0

        for img_idx in range(len(processed_images)):
            current_img = processed_images[img_idx]
            
            # Apply Ken Burns effect if enabled
            if use_ken_burns:
                ken_burns_frames = []
                for frame_idx in range(static_frames):
                    progress = frame_idx / static_frames
                    
                    # Random zoom and pan parameters (set once per image)
                    if frame_idx == 0:
                        zoom_start = np.random.uniform(1.0, 1.1)
                        zoom_end = np.random.uniform(1.05, 1.2)
                        pan_x_start = np.random.uniform(-0.05, 0.05)
                        pan_x_end = np.random.uniform(-0.05, 0.05)
                        pan_y_start = np.random.uniform(-0.05, 0.05)
                        pan_y_end = np.random.uniform(-0.05, 0.05)
                    
                    # Interpolate zoom and pan
                    current_zoom = zoom_start + (zoom_end - zoom_start) * progress
                    current_pan_x = pan_x_start + (pan_x_end - pan_x_start) * progress
                    current_pan_y = pan_y_start + (pan_y_end - pan_y_start) * progress
                    
                    # Apply transformation
                    h, w = current_img.shape[:2]
                    center_x = w // 2 + int(current_pan_x * w)
                    center_y = h // 2 + int(current_pan_y * h)
                    
                    M = cv2.getRotationMatrix2D((center_x, center_y), 0, current_zoom)
                    ken_burns_frame = cv2.warpAffine(current_img, M, (w, h))
                    ken_burns_frames.append(ken_burns_frame)
                
                # Add Ken Burns frames
                frames.extend(ken_burns_frames)
            else:
                # Add static frames
                for _ in range(static_frames):
                    frames.append(current_img.copy())
            
            current_frame += static_frames
            
            # Add transition to next image (except for last image)
            if img_idx < len(processed_images) - 1:
                next_img = processed_images[img_idx + 1]
                
                # Choose transition
                if transition_type == "🌟 Auto Mix (Random)":
                    chosen_transition = np.random.choice(auto_transitions)
                    transition_func = transition_functions[chosen_transition]
                else:
                    transition_func = transition_functions[transition_type]
                
                # Generate transition frames
                for t_frame in range(transition_frames):
                    progress = (t_frame + 1) / transition_frames
                    # Smooth easing function
                    eased_progress = 0.5 * (1 - np.cos(progress * np.pi))
                    
                    try:
                        transition_frame = transition_func(current_img, next_img, eased_progress)
                        frames.append(transition_frame)
                    except Exception as e:
                        # Fallback to fade if transition fails
                        transition_frame = apply_fade_transition(current_img, next_img, eased_progress)
                        frames.append(transition_frame)
                
                current_frame += transition_frames
            
            # Update progress
            progress_bar.progress(0.4 + 0.5 * (current_frame / total_frames), 
                                text=f"Generating frames... {current_frame}/{total_frames}")

        if not frames:
            st.error("❌ No frames generated for slideshow.")
            st.stop()

        progress_bar.progress(0.9, text="Encoding video...")

        # Write MP4 with better error handling
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                temp_path = tmp.name
            
            # Use imageio with better codec settings
            writer = imageio.get_writer(
                temp_path,
                fps=fps,
                codec='libx264',
                ffmpeg_params=[
                    '-pix_fmt', 'yuv420p',
                    '-crf', '23',  # Good quality
                    '-preset', 'medium',
                    '-movflags', '+faststart'  # Web optimization
                ]
            )
            
            for frame in frames:
                writer.append_data(frame)
            writer.close()

            # Read and display result
            with open(temp_path, "rb") as f:
                video_data = f.read()
            
            progress_bar.progress(1.0, text="Complete! 🎉")
            
            st.success("✅ Slideshow ready!")
            
            # Display video info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📹 Total Frames", len(frames))
            with col2:
                st.metric("⏱️ Duration", f"{len(frames)/fps:.1f}s")
            with col3:
                st.metric("📊 File Size", f"{len(video_data)/1024/1024:.1f} MB")
            
            # Video player
            st.video(io.BytesIO(video_data))
            
            # Download button
            st.download_button(
                "⬇️ Download Slideshow (MP4)",
                video_data,
                file_name=f"slideshow_{transition_type.split()[1] if len(transition_type.split()) > 1 else 'custom'}_{out_w}x{out_h}.mp4",
                mime="video/mp4"
            )
            
            # Clean up progress bar
            progress_bar.empty()

        except Exception as e:
            st.error(f"❌ Failed to create slideshow: {e}")
            st.info("💡 Try reducing the number of images or transition duration if you're running out of memory.")

        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass