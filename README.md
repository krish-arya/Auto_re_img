# Auto_re_img
🖼️ Auto Crop, Resize & Compress Web App
This Streamlit app automatically detects a subject (like a person), crops tightly around them, resizes the image to a desired resolution without padding, and compresses it under a specified file size limit.

It uses:

🔍 rembg for background removal & subject detection
📐 Smart cropping & resizing logic
📦 Compression to fit under your desired size (in KB)

🚀 Features
🧠 Automatic Subject Detection using rembg
✂️ Tight Cropping around the subject with optional margin
📏 Resize Without Padding to fit target dimensions
🗜️ JPEG Compression to meet file size requirements
📥 Upload History with clear option
💾 Download Button to save the final image


📦 Requirements
Install the dependencies with:
bash
Copy
Edit
pip install streamlit pillow rembg


▶️ How to Run
Run the app locally using Streamlit:

bash
Copy
Edit
streamlit run app.py
🛠️ App Settings
Output Width / Height (px): Target resolution for the output image
Crop Margin (px): Padding around the detected subject
Max File Size (KB): Final compressed file size limit
Upload History: Keeps track of recent uploads
Clear History: Clears the upload list


📸 How It Works :
Upload any .png, .jpg, or .jpeg image.
Click "🛠 Process Image" to:
Remove background
Crop around the subject with margin
Resize and compress
Download your optimized image with a single click!

✨ Example Use Cases
Optimizing model portfolio images
Preparing e-commerce product shots
Creating content for web upload with file size constraints

📁 Output Format
Format: JPEG


Resolution: Custom (user-defined)
Size: Compressed to stay within your limit (e.g. 800 KB)


🔒 Note
rembg runs locally – no data is sent to external servers.
