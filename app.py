import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# -----------------------------
# Sample Celebrity Image URLs
# Replace with actual public URLs for testing
# -----------------------------
CELEB_IMAGES = {
    "Bipasha Basu": "https://raw.githubusercontent.com/your-username/celebrity-db/main/bipasha_basu.jpg",
    "Aamir Khan": "https://raw.githubusercontent.com/your-username/celebrity-db/main/aamir_khan.jpg",
    "Abhay Deol": "https://raw.githubusercontent.com/your-username/celebrity-db/main/abhay_deol.jpg"
}

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.title("🎬 Bollywood Celebrity Matcher - Test Version")

    st.markdown("## 🖼 Check Dataset Image")
    selected_name = st.selectbox("Pick a celebrity image to view:", list(CELEB_IMAGES.keys()))

    if selected_name:
        url = CELEB_IMAGES[selected_name]
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(img, caption=selected_name, width=300)
            st.success(f"✅ {selected_name} image loaded successfully!")
        except Exception as e:
            st.error(f"❌ Could not load image: {e}")

if __name__ == "__main__":
    main()
