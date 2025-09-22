import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Sample celebrity images (online URLs for testing)
# -----------------------------
CELEB_IMAGES = {
    "Bipasha Basu": "https://raw.githubusercontent.com/Hari99-ai/test-celeb-images/main/bipasha_basu.jpg",
    "Aamir Khan": "https://raw.githubusercontent.com/Hari99-ai/test-celeb-images/main/aamir_khan.jpg",
    "Abhay Deol": "https://raw.githubusercontent.com/Hari99-ai/test-celeb-images/main/abhay_deol.jpg"
}

st.title("🎬 Bollywood Celebrity Matcher - Online Test")

# -----------------------------
# User Input
# -----------------------------
option = st.radio("Choose your input:", ["📁 Upload Image", "📷 Take a Selfie", "🖼 Pick Online Test Image"])
user_image = None

if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Choose your image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file:
        user_image = Image.open(uploaded_file).convert("RGB")
        st.image(user_image, caption="Your Image", width=250)

elif option == "📷 Take a Selfie":
    camera_file = st.camera_input("Take a selfie")
    if camera_file:
        user_image = Image.open(camera_file).convert("RGB")
        st.image(user_image, caption="Your Selfie", width=250)

elif option == "🖼 Pick Online Test Image":
    celeb_name = st.selectbox("Pick a celebrity image:", list(CELEB_IMAGES.keys()))
    if celeb_name:
        response = requests.get(CELEB_IMAGES[celeb_name])
        user_image = Image.open(BytesIO(response.content)).convert("RGB")
        st.image(user_image, caption=celeb_name, width=250)

# -----------------------------
# Test Matching (Fake embeddings for demo)
# -----------------------------
if user_image is not None and st.button("🔍 Find My Celebrity Match!"):
    st.info("🔄 Computing similarity (demo)...")
    
    # For testing purposes, random similarity scores
    similarities = {name: np.random.randint(50, 100) for name in CELEB_IMAGES.keys()}
    sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    st.markdown("## 🎭 Your Matches")
    for rank, (name, score) in enumerate(sorted_matches, 1):
        st.metric(f"#{rank} Match: {name}", f"{score}% similarity")

