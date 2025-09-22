import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Bollywood Celebrity Matcher - Online Image Test")

# -----------------------------
# User Input Options
# -----------------------------
option = st.radio("Choose your input:", ["📁 Upload Image", "📷 Take a Selfie", "🌐 Use Online Image URL"])
user_image = None

# Upload from local
if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Choose your image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file:
        user_image = Image.open(uploaded_file).convert("RGB")
        st.image(user_image, caption="Your Image", width=250)

# Take a selfie
elif option == "📷 Take a Selfie":
    camera_file = st.camera_input("Take a selfie")
    if camera_file:
        user_image = Image.open(camera_file).convert("RGB")
        st.image(user_image, caption="Your Selfie", width=250)

# Online image via URL
elif option == "🌐 Use Online Image URL":
    url = st.text_input("Enter image URL")
    if url:
        try:
            response = requests.get(url)
            user_image = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(user_image, caption="Online Image", width=250)
        except Exception as e:
            st.error(f"❌ Failed to load image from URL: {e}")

# -----------------------------
# Demo Matching (Random for testing)
# -----------------------------
if user_image is not None and st.button("🔍 Find My Celebrity Match!"):
    st.info("🔄 Computing similarity (demo)...")
    
    # Fake celeb dataset for demo
    CELEB_IMAGES = ["Bipasha Basu", "Aamir Khan", "Abhay Deol"]
    similarities = {name: np.random.randint(50, 100) for name in CELEB_IMAGES}
    sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    st.markdown("## 🎭 Your Matches")
    for rank, (name, score) in enumerate(sorted_matches, 1):
        st.metric(f"#{rank} Match: {name}", f"{score}% similarity")

