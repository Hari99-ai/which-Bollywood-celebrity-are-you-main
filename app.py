import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.title("🎬 Bollywood Celebrity Matcher - Test Image")

# GitHub raw URL of Aamir Khan image
image_url = "https://raw.githubusercontent.com/Hari99-ai/which-Bollywood-celebrity-are-you-main/main/data/ami/Aamir.40.jpg"

# Fetch image from URL
try:
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    st.image(img, caption="Aamir Khan (Test)", width=250)
    st.success("✅ Image loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load image: {e}")
