import streamlit as st
from PIL import Image, ImageOps, ImageColor
from rembg import remove
import io
import zipfile
import cv2
import numpy as np
from ultralytics import YOLO

# ✅ Set page config FIRST
st.set_page_config(
    page_title="Enhanced Smart Zoom with AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize YOLO model once
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')  # Downloads if not present

def smart_crop_preserve_subject(model, img: Image.Image, target_size: tuple, zoom_factor: float = 1.3, use_inpainting=False, bg_color=(0,0,0)):
    img_cv = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    results = model.predict(img_cv, verbose=False)
    persons = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for box, cls in zip(boxes, classes):
            if cls == 0:  # person
                persons.append(box)
    
    if persons:
        largest = max(persons, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        l, t, r, b = map(int, largest)
    else:
        input_buffer = io.BytesIO()
        img.save(input_buffer, format="PNG")
        fg_bytes = remove(input_buffer.getvalue())
        fg_img = Image.open(io.BytesIO(fg_bytes)).convert("RGBA")
        alpha = fg_img.split()[-1]
        bbox = alpha.getbbox() or (0, 0, img.width, img.height)
        l, t, r, b = bbox

    center_x = (l + r) // 2
    center_y = (t + b) // 2
    box_w, box_h = r - l, b - t

    new_w = min(int(box_w * zoom_factor), img.width)
    new_h = min(int(box_h * zoom_factor), img.height)

    left = max(center_x - new_w//2, 0)
    right = min(center_x + new_w//2, img.width)
    upper = max(center_y - new_h//2, 0)
    lower = min(center_y + new_h//2, img.height)

    cropped = img.crop((left, upper, right, lower))
    return resize_and_pad(cropped, target_size, bg_color, use_inpainting)

def resize_and_pad(img: Image.Image, target_size: tuple, bg_color=(0,0,0), use_inpainting=False):
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]

    if img_ratio > target_ratio:
        new_w = target_size[0]
        new_h = int(new_w / img_ratio)
    else:
        new_h = target_size[1]
        new_w = int(new_h * img_ratio)

    resized = img.resize((new_w, new_h), Image.LANCZOS)

    if new_w == target_size[0] and new_h == target_size[1]:
        return resized

    if use_inpainting:
        resized_cv = cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)
        padded = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        x_start = (target_size[0] - new_w) // 2
        y_start = (target_size[1] - new_h) // 2
        padded[y_start:y_start+new_h, x_start:x_start+new_w] = resized_cv

        mask = np.zeros(padded.shape[:2], dtype=np.uint8)
        mask[:y_start, :] = 255
        mask[y_start+new_h:, :] = 255
        mask[:, :x_start] = 255
        mask[:, x_start+new_w:] = 255

        inpainted = cv2.inpaint(padded, mask, 3, cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    else:
        delta_w = target_size[0] - new_w
        delta_h = target_size[1] - new_h
        padding = (delta_w//2, delta_h//2, delta_w - delta_w//2, delta_h - delta_h//2)
        return ImageOps.expand(resized, padding, fill=bg_color)

def main():
    model = load_yolo_model()

    st.title("📸 Smart AI Zoom & Crop Tool")
    st.markdown("Upload images, preview before & after, and download enhanced versions.")

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        width = st.number_input("Output Width", 100, 2000, 512, step=10)
        height = st.number_input("Output Height", 100, 2000, 512, step=10)
        zoom_factor = st.slider("Zoom Factor", 1.0, 2.0, 1.3, step=0.1)
        use_inpainting = st.checkbox("Use AI Background Fill", False)
        bg_color = st.color_picker("Fallback Background Color", "#000000")

    files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if files:
        st.subheader("📂 Uploaded Previews")
        cols = st.columns(min(4, len(files)))
        for i, file in enumerate(files):
            with cols[i % len(cols)]:
                st.image(file, caption=file.name, use_column_width=True)

        if st.button("🚀 Process All Images"):
            bg_rgb = ImageColor.getrgb(bg_color)
            processed = []
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for file in files:
                    img = Image.open(file).convert('RGB')
                    result = smart_crop_preserve_subject(
                        model, img, (width, height), zoom_factor, use_inpainting, bg_rgb
                    )

                    # Show preview of processed
                    st.image(result, caption=f"Processed: {file.name}", use_column_width=True)

                    # Save to zip
                    img_byte_arr = io.BytesIO()
                    result.save(img_byte_arr, format="JPEG", quality=95)
                    zip_file.writestr(f"{file.name.rsplit('.', 1)[0]}_processed.jpg", img_byte_arr.getvalue())
                    processed.append(result)

            st.success("✅ Processing complete!")
            st.download_button(
                label="📦 Download All as ZIP",
                data=zip_buffer.getvalue(),
                file_name="processed_images.zip",
                mime="application/zip"
            )

if __name__ == "__main__":
    main()
