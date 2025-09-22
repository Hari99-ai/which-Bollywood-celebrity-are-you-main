import os
import streamlit as st
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from PIL import Image
from io import BytesIO

# -----------------------------
# Setup
# -----------------------------
CELEB_DB_FOLDER = "celebrity_db"
os.makedirs(CELEB_DB_FOLDER, exist_ok=True)
FOLDER_ID = "1qDeCZPwzsmvXwvfolkqcXQdWOH-wXYIr"  # your Drive folder ID

# -----------------------------
# Authenticate with Google Drive
# -----------------------------
def gdrive_login():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # opens browser for first-time auth
    return GoogleDrive(gauth)

# -----------------------------
# Download image from Google Drive API
# -----------------------------
def get_images(drive, folder_id):
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    images = []
    for file in file_list:
        # Skip if already downloaded
        local_path = os.path.join(CELEB_DB_FOLDER, file['title'])
        if not os.path.exists(local_path):
            st.write(f"📥 Downloading {file['title']}...")
            file.GetContentFile(local_path)
        images.append(local_path)
    return images

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.title("🎬 Bollywood Celebrity Matcher")

    drive = gdrive_login()
    celeb_images = get_images(drive, FOLDER_ID)

    st.success(f"✅ {len(celeb_images)} celebrity images available!")

    # Show any image from dataset
    selected_file = st.selectbox("Pick a celebrity image to view:", celeb_images)
    if selected_file:
        img = Image.open(selected_file).convert("RGB")
        st.image(img, caption=os.path.basename(selected_file), width=250)

if __name__ == "__main__":
    main()
