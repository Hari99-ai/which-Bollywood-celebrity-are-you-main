import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# -----------------------------
# Sample celebrity images online (replace with real URLs)
# -----------------------------
CELEB_IMAGES = {
    "Aamir Khan": "https://raw.githubusercontent.com/hari99-ai/test-images/main/aamir_khan.jpg",
    "Abhay Deol": "https://raw.githubusercontent.com/hari99-ai/test-images/main/abhay_deol.jpg",
    "Bipasha Basu": "https://raw.githubusercontent.com/hari99-ai/test-images/main/bipasha_basu.jpg"
}

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.title("🎬 Bollywood Celebrity Matcher (Online Test)")

    st.markdown("## 🖼 View Dataset Image")
    celeb_name = st.selectbox("Pick a celebrity image to view:", list(CELEB_IMAGES.keys()))

    if celeb_name:
        url = CELEB_IMAGES[celeb_name]
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(img, caption=celeb_name, width=250)
            st.success(f"✅ Loaded image of {celeb_name} successfully!")
        except Exception as e:
            st.error(f"❌ Could not load image: {e}")

    st.markdown("---")
    st.markdown("### 📸 Test Upload or Camera Input")
    upload_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png", "webp"])
    camera_file = st.camera_input("Take a selfie")

    if upload_file:
        img = Image.open(upload_file).convert("RGB")
        st.image(img, caption="Uploaded Image", width=250)
    elif camera_file:
        img = Image.open(camera_file).convert("RGB")
        st.image(img, caption="Camera Image", width=250)

if __name__ == "__main__":
    main()
