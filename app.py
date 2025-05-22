import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io, zipfile, cv2, numpy as np
from rembg import remove
from ultralytics import YOLO
from typing import Tuple, Optional, Union
import torch
from transformers import CLIPProcessor, CLIPModel
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# ========== Page Configuration ==========
st.set_page_config(
    page_title="AI Cropper + Brand Generator",
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# ========== Model Loading ==========
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n-seg.pt')

@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return processor, model

# ========== Utility Functions ==========
def compute_center_of_bbox(bbox: Tuple[int]) -> Tuple[int, int]:
    return (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2

def enhanced_subject_detection(model, img: Image.Image) -> Optional[Tuple[int]]:
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
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

def smart_crop_with_headspace(img: Image.Image, bbox: Tuple[int], target_size: Tuple[int], zoom: float,
                               headspace_top: Union[int, float], headspace_bottom: Union[int, float],
                               use_percent: bool = False) -> Image.Image:
    img_w, img_h = img.size
    t_width, t_height = target_size
    subject_width = bbox[2] - bbox[0]
    subject_height = bbox[3] - bbox[1]
    zoomed_width = subject_width * zoom
    zoomed_height = subject_height * zoom
    scale = min(t_width / zoomed_width, t_height / zoomed_height)
    crop_width = int(t_width / scale)
    crop_height = int(t_height / scale)
    top_extra = int(headspace_top * subject_height / 100) if use_percent else int(headspace_top)
    bottom_extra = int(headspace_bottom * subject_height / 100) if use_percent else int(headspace_bottom)
    cx, cy = compute_center_of_bbox(bbox)
    top = max(0, cy - crop_height // 2 - top_extra)
    bottom = min(img_h, cy + crop_height // 2 + bottom_extra)
    left = max(0, cx - crop_width // 2)
    right = min(img_w, cx + crop_width // 2)
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

def apply_branding(img: Image.Image, logo: Optional[Image.Image], **kwargs) -> Image.Image:
    composite = img.convert("RGBA")

    if logo:
        logo = logo.convert("RGBA")
        logo_width = int((kwargs["logo_scale"] / 100) * composite.width)
        logo_height = int(logo_width * (logo.height / logo.width))
        logo_resized = logo.resize((logo_width, logo_height), Image.LANCZOS)
        x_px = int((kwargs["x_offset"] / 100) * (composite.width - logo_width))
        y_px = int((kwargs["y_offset"] / 100) * (composite.height - logo_height))
        if kwargs["add_padding"]:
            new_w = composite.width + 2 * kwargs["padding"]
            new_h = composite.height + 2 * kwargs["padding"]
            padded = Image.new("RGBA", (new_w, new_h), kwargs["padding_color"])
            padded.paste(composite, (kwargs["padding"], kwargs["padding"]))
            x_px += kwargs["padding"]
            y_px += kwargs["padding"]
            composite = padded
        composite.paste(logo_resized, (x_px, y_px), logo_resized)

    if kwargs["add_text"] and kwargs["text"]:
        draw = ImageDraw.Draw(composite)
        try:
            font = ImageFont.truetype("arial.ttf", kwargs["font_size"])
        except:
            font = ImageFont.load_default()
        tx = int((kwargs["text_x"] / 100) * composite.width)
        ty = int((kwargs["text_y"] / 100) * composite.height)
        draw.text((tx, ty), kwargs["text"], fill=kwargs["text_color"], font=font)

    return composite.convert("RGB")

# ========== Load Models ==========
model = load_yolo_model()
clip_processor, clip_model = load_clip_model()

# ========== UI Layout ==========
st.title("🎯 AI-Powered Smart Cropper + Brand Generator")

uploaded_files = st.file_uploader("📸 Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
logo_file = st.sidebar.file_uploader("🏷️ Upload Logo (Optional)", type=["png"])

# Show preview
if uploaded_files:
    st.subheader("🔍 Uploaded Image Previews")
    cols = st.columns(min(4, len(uploaded_files)))
    for i, file in enumerate(uploaded_files):
        img = Image.open(file)
        cols[i % len(cols)].image(img, use_container_width=True, caption=file.name)

# Settings
with st.sidebar:
    st.header("📏 Crop Settings")
    target_width = st.number_input("Output Width", 512, 4096, 1080)
    target_height = st.number_input("Output Height", 512, 4096, 1350)
    zoom_factor = st.slider("Zoom Level", 0.5, 3.0, 1.2)
    use_percent = st.checkbox("Headspace in Percent", True)
    top_space = st.number_input("Top Headspace", 0, 1000, 10)
    bottom_space = st.number_input("Bottom Headspace", 0, 1000, 10)
    max_size_kb = st.number_input("Max File Size (KB)", 100, 5000, 800)

    st.header("🛠 Branding Settings")
    logo_scale = st.slider("Logo Size (% of width)", 5, 50, 15)
    x_offset = st.slider("Logo X Position (%)", 0, 100, 50)
    y_offset = st.slider("Logo Y Position (%)", 0, 100, 90)
    add_text = st.checkbox("Add Text")
    text = st.text_input("Overlay Text", "Your Brand Message")
    font_size = st.slider("Font Size", 10, 150, 40)
    text_color = st.color_picker("Text Color", "#FFFFFF")
    text_x = st.slider("Text X Position (%)", 0, 100, 5)
    text_y = st.slider("Text Y Position (%)", 0, 100, 5)
    add_padding = st.checkbox("Add Padding")
    padding = st.slider("Padding (px)", 0, 300, 50)
    padding_color = st.color_picker("Padding Color", "#000000")

# ========== Main Logic ==========
if uploaded_files and st.button("🚀 Process Images"):
    results = []
    logo = Image.open(logo_file).convert("RGBA") if logo_file else None

    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        bbox = enhanced_subject_detection(model, img)
        if bbox is None or len(bbox) == 0:
            bbox = (img.width // 4, img.height // 4, 3 * img.width // 4, 3 * img.height // 4)
        cropped = smart_crop_with_headspace(
            img, bbox, (target_width, target_height), zoom_factor,
            top_space, bottom_space, use_percent
        )
        branded = apply_branding(
            cropped, logo,
            logo_scale=logo_scale, x_offset=x_offset, y_offset=y_offset,
            add_text=add_text, text=text, font_size=font_size,
            text_color=text_color, text_x=text_x, text_y=text_y,
            add_padding=add_padding, padding=padding, padding_color=padding_color
        )
        buffer = optimize_image(branded, max_size_kb)
        results.append((file.name, branded, buffer))

    # Show results in columns (preview)
    st.subheader("🎨 Branded Output Preview")
    preview_cols = st.columns(min(4, len(results)))
    for i, (name, img, _) in enumerate(results):
        preview_cols[i % len(preview_cols)].image(img, caption=name, use_container_width=True)

    # Individual download buttons
    for name, img, buf in results:
        st.download_button(f"⬇ Download {name}", data=buf.getvalue(), file_name=f"branded_{name}", mime="image/jpeg")

    # ZIP download
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zipf:
        for name, _, buf in results:
            zipf.writestr(f"branded_{name}", buf.getvalue())
    st.download_button("📦 Download All as ZIP", data=zip_buf.getvalue(), file_name="branded_images.zip", mime="application/zip")