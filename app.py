import os
import gdown
import zipfile
import streamlit as st
from PIL import Image

# ---------------------------
# Step 1: Setup local folder
# ---------------------------
CELEB_DB_FOLDER = "celebrity_db"
os.makedirs(CELEB_DB_FOLDER, exist_ok=True)

# ---------------------------
# Step 2: Download ZIP from Google Drive
# ---------------------------
# ⚠️ Replace with your ZIP file's Google Drive file ID
# Example link: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
drive_zip_url = "https://drive.google.com/uc?id=YOUR_FILE_ID"

zip_path = "celebrity_db.zip"

if not os.path.exists(zip_path):
    st.info("📥 Downloading celebrity database ZIP from Google Drive...")
    gdown.download(drive_zip_url, zip_path, quiet=False)

# Extract only once
if not os.listdir(CELEB_DB_FOLDER):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(CELEB_DB_FOLDER)

# ---------------------------
# Step 3: Streamlit app
# ---------------------------
def main():
    st.title("🎬 Bollywood Celebrity Matcher")
    st.success("✅ Ready! Celebrity database extracted from Google Drive ZIP.")

    # Show options
    option = st.radio("Choose your input:", ["📁 Upload Image", "📷 Take a Selfie", "🖼 Check Dataset Image"])
    image = None

    if option == "📁 Upload Image":
        uploaded_file = st.file_uploader("Choose your image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, width=200)
            except Exception as e:
                st.error(f"❌ Could not open uploaded file. Error: {e}")

    elif option == "📷 Take a Selfie":
        camera_file = st.camera_input("Take a photo")
        if camera_file is not None:
            try:
                image = Image.open(camera_file).convert("RGB")
                st.image(image, width=200)
            except Exception as e:
                st.error(f"❌ Could not open captured photo. Error: {e}")

    elif option == "🖼 Check Dataset Image":
        celeb_list = os.listdir(CELEB_DB_FOLDER)
        if celeb_list:
            selected_celeb = st.selectbox("Pick a celebrity", celeb_list)
            img_path = os.path.join(CELEB_DB_FOLDER, selected_celeb)
            try:
                image = Image.open(img_path).convert("RGB")
                st.image(image, width=200, caption=selected_celeb)
            except Exception as e:
                st.error(f"❌ Could not load dataset image. Error: {e}")
        else:
            st.warning("⚠️ No celebrity images found in the database.")

    if image is not None:
        st.success("✅ Image loaded successfully! Ready for matching...")

if __name__ == "__main__":
    main()
