import streamlit as st
from PIL import Image
from rembg import remove
import io
import zipfile

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

# Function to resize by smart cropping (no padding, no background)
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
    rgb_img = img.convert('RGB')
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
    # Collapse sidebar by default and set page config
    st.set_page_config(
        page_title="Batch Image Cropper & Resizer",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
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

    
    st.title("Automated Resizer & Compress Around Model")
    # Prompt user to open the sidebar
    st.info("💡 Click the sidebar toggle (▸) to open settings and adjust dimensions, margin, or file size.")
    st.subheader("Start by uploading images to process.")

    st.sidebar.header("Settings")
    width = st.sidebar.number_input("Output Width (px)", min_value=100, max_value=5000, value=1200)
    height = st.sidebar.number_input("Output Height (px)", min_value=100, max_value=5000, value=1800)
    margin = st.sidebar.number_input("Crop Margin (px)", min_value=0, max_value=500, value=20)
    max_kb = st.sidebar.number_input("Max File Size (KB)", min_value=10, max_value=10000, value=800)
    history_clear = st.sidebar.button("🧼 Clear History")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = []
    if "zip_bytes" not in st.session_state:
        st.session_state.zip_bytes = None

    if history_clear:
        st.session_state.history.clear()

    uploaded = st.file_uploader("Choose images (PNG/JPEG)...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded:
        st.session_state.uploaded_files = uploaded
        for file in uploaded:
            if file.name not in st.session_state.history:
                st.session_state.history.append(file.name)
                
    st.sidebar.subheader("Upload History")
    for i, name in enumerate(reversed(st.session_state.history), 1):
        st.sidebar.text(f"{i}. {name}")

    if st.session_state.uploaded_files:
        st.subheader("Uploaded Images Preview")
        cols = st.columns(3)
        for idx, file in enumerate(st.session_state.uploaded_files):
            file_bytes = file.read()
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            size_kb = len(file_bytes) / 1024
            with cols[idx % 3]:
                st.image(img, caption=f"{file.name}\n{img.width}x{img.height}px | {size_kb:.1f}KB", use_container_width=True)
            file.seek(0)

        if st.button("🛠️ Process All Images"):
            st.session_state.processed_data = []
            for file in st.session_state.uploaded_files:
                file_bytes = file.read()
                img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                original_dims = (img.width, img.height)
                original_kb = len(file_bytes) / 1024
                processed = crop_around_subject(img, margin=margin) if (img.width > width or img.height > height) else img
                resized = resize_with_crop(processed, (width, height))
                out_bytes, out_kb, out_q = compress_image(resized, max_kb)
                st.session_state.processed_data.append(
                    (file.name, resized, out_bytes, original_dims, original_kb, resized.size, out_kb, out_q)
                )
                file.seek(0)

    if st.session_state.processed_data:
        st.subheader("Processed Images Preview")
        for name, pil_img, _, orig_dims, orig_kb, proc_dims, proc_kb, quality in st.session_state.processed_data:
            st.image(pil_img, caption=(
                f"{name} | {orig_dims[0]}x{orig_dims[1]} → {proc_dims[0]}x{proc_dims[1]} px | "
                f"{orig_kb:.1f} KB → {proc_kb:.1f} KB | Q={quality}"
            ), use_container_width=True)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, _, out_bytes, *_ in st.session_state.processed_data:
                zf.writestr(f"processed_{name.split('.')[0]}.jpg", out_bytes)
        zip_buffer.seek(0)
        st.session_state.zip_bytes = zip_buffer.getvalue()

    if st.session_state.zip_bytes:
        st.subheader("Download Processed Images")
        st.download_button("Download ZIP", data=st.session_state.zip_bytes,
                       file_name="processed_images.zip", mime="application/zip")

if __name__ == "__main__":
    main()
