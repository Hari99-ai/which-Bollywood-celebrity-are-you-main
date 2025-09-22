import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Sample public celebrity images
CELEB_IMAGES = {
    "Bipasha Basu": "https://raw.githubusercontent.com/hari/celebrity-db/main/bipasha_basu.jpg",

}

st.title("🎬 Bollywood Celebrity Matcher - Test Version")

selected_name = st.selectbox("Pick a celebrity image to view:", list(CELEB_IMAGES.keys()))
if selected_name:
    try:
        response = requests.get(CELEB_IMAGES[selected_name])
        img = Image.open(BytesIO(response.content)).convert("RGB")
        st.image(img, caption=selected_name, width=300)
        st.success(f"✅ {selected_name} image loaded successfully!")
    except Exception as e:
        st.error(f"❌ Could not load image: {e}")

