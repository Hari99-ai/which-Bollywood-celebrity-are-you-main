import os
import streamlit as st
from PIL import Image

# Optional import; may need to install
try:
    import gdown
except ImportError:
    st.error("gdown not installed. `pip install gdown` and refresh.")
    raise

# -----------------------------
# Setup constants
# -----------------------------
CELEB_DB_FOLDER = "celebrity_db"
os.makedirs(CELEB_DB_FOLDER, exist_ok=True)

DRV_FOLDER_URL = "https://drive.google.com/drive/folders/1qDeCZPwzsmvXwvfolkqcXQdWOH-wXYIr?usp=drive_link"

# -----------------------------
# Function to download folder via gdown
# -----------------------------
def download_celebrity_folder(url, target_folder):
    try:
        st.info("📥 Trying to download folder via gdown.download_folder...")
        # remaining_ok=True may allow more files if folder > 50
        gdown.download_folder(url, output=target_folder, quiet=False, use_cookies=False, remaining_ok=True)
        return True
    except Exception as e:
        st.warning(f"⚠️ download_folder failed: {e}")
        return False

# -----------------------------
# Main Logic
# -----------------------------
def ensure_celebrity_folder():
    # If folder already has image files, assume it's ready
    existing = [f for f in os.listdir(CELEB_DB_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    if existing:
        return True

    # Try download via gdown
    ok = download_celebrity_folder(DRV_FOLDER_URL, CELEB_DB_FOLDER)
    if ok:
        st.success("✅ Celebrity folder downloaded successfully.")
        return True
    else:
        st.error("❌ Could not download celebrity folder via gdown. Please check:")
        st.text("- The folder is set to 'Anyone with link' (Viewer)")
        st.text("- The folder is not too large (or try splitting it)")
        st.text("- Network / firewall restrictions")
        return False

# -----------------------------
# App
# -----------------------------
def main():
    st.title("🎬 Bollywood Celebrity Matcher")

    # Ensure dataset is present
    data_ready = ensure_celebrity_folder()

    if not data_ready:
        st.stop()

    st.success(f"✅ {len(os.listdir(CELEB_DB_FOLDER))} files found in celebrity_db.")

    # UI input options
    option = st.radio("Choose your input:", ["📁 Upload Image", "📷 Take a Selfie", "🖼 Check Dataset Image"])
    image = None

    if option == "📁 Upload Image":
        uploaded_file = st.file_uploader("Choose your image", type=["jpg","jpeg","png","webp"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded", width=200)

    elif option == "📷 Take a Selfie":
        camera_file = st.camera_input("Take a Selfie")
        if camera_file:
            image = Image.open(camera_file).convert("RGB")
            st.image(image, caption="Selfie", width=200)

    elif option == "🖼 Check Dataset Image":
        celebs = [f for f in os.listdir(CELEB_DB_FOLDER) if f.lower().endswith(('.jpg','jpeg','png','webp'))]
        if celebs:
            selected = st.selectbox("Select a celebrity image", celebs)
            img_path = os.path.join(CELEB_DB_FOLDER, selected)
            try:
                image = Image.open(img_path).convert("RGB")
                st.image(image, caption=selected, width=200)
            except Exception as e:
                st.error(f"❌ Could not load image {selected}: {e}")
        else:
            st.warning("⚠️ No valid image files in celebrity_db.")

    if image:
        st.success("✅ Image loaded! You can now run matching logic.")

if __name__ == "__main__":
    main()
