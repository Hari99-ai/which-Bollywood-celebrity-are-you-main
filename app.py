import os
import streamlit as st
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# -----------------------------
# Setup
# -----------------------------
CELEB_DB_FOLDER = "celebrity_db"
os.makedirs(CELEB_DB_FOLDER, exist_ok=True)

# Your folder ID (from link: https://drive.google.com/drive/folders/<FOLDER_ID>)
FOLDER_ID = "1qDeCZPwzsmvXwvfolkqcXQdWOH-wXYIr"

# -----------------------------
# Authenticate
# -----------------------------
def gdrive_login():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()   # will open browser for auth
    return GoogleDrive(gauth)

# -----------------------------
# Download all files from folder
# -----------------------------
def download_folder(drive, folder_id, target_folder):
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    for file in file_list:
        fname = os.path.join(target_folder, file['title'])
        if not os.path.exists(fname):
            st.write(f"📥 Downloading {file['title']}...")
            file.GetContentFile(fname)

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.title("🎬 Bollywood Celebrity Matcher")

    if st.button("📥 Download Celebrity DB from Drive"):
        try:
            drive = gdrive_login()
            download_folder(drive, FOLDER_ID, CELEB_DB_FOLDER)
            st.success(f"✅ Download complete! {len(os.listdir(CELEB_DB_FOLDER))} files in celebrity_db.")
        except Exception as e:
            st.error(f"❌ Failed: {e}")

    # Show existing files
    if os.listdir(CELEB_DB_FOLDER):
        st.write("Celebrity images available:")
        st.write(os.listdir(CELEB_DB_FOLDER)[:10])  # preview first 10

if __name__ == "__main__":
    main()
