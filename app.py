import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.title("🎬 Bollywood Celebrity Image Test")

# GitHub raw image URL
url = "https://raw.githubusercontent.com/Hari99-ai/which-Bollywood-celebrity-are-you-main/main/data/ami/Aamir.44.jpg"

try:
    response = requests.get(url)
    response.raise_for_status()  # check for errors
    img = Image.open(BytesIO(response.content)).convert("RGB")
    st.image(img, caption="Aamir Khan", width=300)
    st.success("✅ Image loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load image: {e}")
