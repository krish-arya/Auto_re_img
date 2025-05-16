import streamlit as st
from PIL import Image, ImageOps
from rembg import remove
import io

# Function to detect and crop around the model using background removal
def crop_around_subject(img: Image.Image, margin: int = 20):
    input_buffer = io.BytesIO()
    img.save(input_buffer, format="PNG")
    fg_bytes = remove(input_buffer.getvalue())
    fg_img = Image.open(io.BytesIO(fg_bytes)).convert("RGBA")

    alpha = fg_img.split()[-1]
    bbox = alpha.getbbox()
    if not bbox:
        bbox = (0, 0, img.width, img.height)

    left = max(bbox[0] - margin, 0)
    upper = max(bbox[1] - margin, 0)
    right = min(bbox[2] + margin, img.width)
    lower = min(bbox[3] + margin, img.height)
    return img.crop((left, upper, right, lower))

# ✅ Function to resize by smart cropping (no padding, no background)
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

# Function to compress image under max_kb
def compress_image(img: Image.Image, max_kb: int):
    rgb_img = img.convert('RGB')  # Always flatten to RGB
    buffer = io.BytesIO()
    quality = 95
    while quality >= 20:
        buffer.seek(0)
        buffer.truncate()
        rgb_img.save(buffer, format='JPEG', quality=quality, optimize=True)
        size_kb = buffer.tell() / 1024
        if size_kb <= max_kb:
            buffer.seek(0)
            return buffer.getvalue(), size_kb, quality
        quality -= 5
    buffer.seek(0)
    return buffer.getvalue(), size_kb, quality

# Streamlit UI
def main():
    st.set_page_config(page_title="Image Cropper & Resizer", layout="wide")
    st.title("Auto Crop, Resize & Compress Around Model (No Padding)")

    st.sidebar.header("Settings")
    width = st.sidebar.number_input("Output Width (px)", min_value=100, max_value=5000, value=1200)
    height = st.sidebar.number_input("Output Height (px)", min_value=100, max_value=5000, value=1800)
    margin = st.sidebar.number_input("Crop Margin (px)", min_value=0, max_value=500, value=20)
    max_kb = st.sidebar.number_input("Max File Size (KB)", min_value=10, max_value=10000, value=800)
    history_clear = st.sidebar.button("🧼 Clear History")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "uploaded_bytes" not in st.session_state:
        st.session_state.uploaded_bytes = None
    if "final_bytes" not in st.session_state:
        st.session_state.final_bytes = None

    if history_clear:
        st.session_state.history.clear()

    st.sidebar.subheader("Upload History")
    for i, name in enumerate(reversed(st.session_state.history), 1):
        st.sidebar.text(f"{i}. {name}")

    uploaded = st.file_uploader("Choose an image (PNG/JPEG)...", type=["png", "jpg", "jpeg"])
    if uploaded:
        data = uploaded.read()
        st.session_state.uploaded_bytes = data
        if uploaded.name not in st.session_state.history:
            st.session_state.history.append(uploaded.name)

    if st.session_state.uploaded_bytes:
        img = Image.open(io.BytesIO(st.session_state.uploaded_bytes)).convert("RGB")
        orig_kb = len(st.session_state.uploaded_bytes) / 1024
        st.subheader("Original Image")
        st.image(img, use_column_width=True)
        st.write(f"Original Dimensions: {img.width}×{img.height} px")
        st.write(f"Original File Size: {orig_kb:.1f} KB")
        st.write(f"Target Dimensions: {width}×{height} px, Max Size: {max_kb} KB")

        if st.button("🛠 Process Image"):
            processed = crop_around_subject(img, margin=margin) if (img.width > width or img.height > height) else img
            resized = resize_with_crop(processed, (width, height))  # ✅ Updated here
            compressed_bytes, final_kb, used_quality = compress_image(resized, max_kb)
            st.session_state.final_bytes = compressed_bytes
            st.session_state.final_info = (resized.width, resized.height, final_kb, used_quality)

    if st.session_state.final_bytes:
        w, h, size_kb, quality = st.session_state.final_info
        st.subheader("Final Image")
        st.image(st.session_state.final_bytes, use_column_width=True)
        st.write(f"Final Dimensions: {w}×{h} px")
        st.write(f"Final File Size: {size_kb:.1f} KB (Quality: {quality})")
        st.download_button(
            "Download Final Image",
            data=st.session_state.final_bytes,
            file_name="final_image.jpg",
            mime="image/jpeg"
        )

if __name__ == "__main__":
    main()