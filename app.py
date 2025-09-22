import os
import gdown
import streamlit as st
from PIL import Image

# ---------------------------
# Step 1: Setup local folder
# ---------------------------
CELEB_DB_FOLDER = "celebrity_db"
os.makedirs(CELEB_DB_FOLDER, exist_ok=True)

# ---------------------------
# Step 2: Google Drive folder
# ---------------------------
# You shared this folder link
drive_folder_url = "https://drive.google.com/drive/folders/1qDeCZPwzsmvXwvfolkqcXQdWOH-wXYIr"

# Convert folder link → gdown format
gdown.download_folder(drive_folder_url, output=CELEB_DB_FOLDER, quiet=False, use_cookies=False)

# ---------------------------
# Step 3: Streamlit app
# ---------------------------
def main():
    st.title("🎬 Bollywood Celebrity Matcher")
    st.success("✅ Ready! Celebrity database synced from Google Drive.")

    # Show options
    option = st.radio("Choose your input:", ["📁 Upload Image", "📷 Take a Selfie", "🖼 Check Dataset Image"])
    image = None

    if option == "📁 Upload Image":
        uploaded_file = st.file_uploader("Choose your image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, width=200)

    elif option == "📷 Take a Selfie":
        camera_file = st.camera_input("Take a photo")
        if camera_file is not None:
            image = Image.open(camera_file).convert("RGB")
            st.image(image, width=200)

    elif option == "🖼 Check Dataset Image":
        celeb_list = os.listdir(CELEB_DB_FOLDER)
        selected_celeb = st.selectbox("Pick a celebrity", celeb_list)
        if selected_celeb:
            img_path = os.path.join(CELEB_DB_FOLDER, selected_celeb)
            try:
                image = Image.open(img_path).convert("RGB")
                st.image(image, width=200, caption=selected_celeb)
            except Exception as e:
                st.error(f"❌ Could not load image: {e}")

    if image is not None:
        st.success("✅ Image loaded successfully! Ready for matching...")

if __name__ == "__main__":
    main()
