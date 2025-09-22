import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace
import os

st.set_page_config(page_title="Bollywood Celebrity Matcher", page_icon="🎬", layout="wide")
st.title("🎬 Bollywood Celebrity Matcher - GitHub Test")

# -----------------------------
# Celebrity dataset URLs (from GitHub)
# -----------------------------
CELEB_DATA = {
    "Aamir Khan": "https://raw.githubusercontent.com/Hari99-ai/which-Bollywood-celebrity-are-you-main/main/data/ami/Aamir.40.jpg",
    "Abhay Deol": "https://raw.githubusercontent.com/Hari99-ai/which-Bollywood-celebrity-are-you-main/main/data/ami/Abhay Deol.153.jpg"
}

# -----------------------------
# User input
# -----------------------------
option = st.radio("Choose your input:", ["📁 Upload Image", "📷 Take a Selfie", "🌐 Use Online Image URL"])
user_image_path = None

# Upload
if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Choose your image", type=["jpg","jpeg","png","webp"])
    if uploaded_file:
        user_image = Image.open(uploaded_file).convert("RGB")
        user_image_path = "user_image.jpg"
        user_image.save(user_image_path)
        st.image(user_image, caption="Your Image", width=250)

# Selfie
elif option == "📷 Take a Selfie":
    camera_file = st.camera_input("Take a selfie")
    if camera_file:
        user_image = Image.open(camera_file).convert("RGB")
        user_image_path = "user_image.jpg"
        user_image.save(user_image_path)
        st.image(user_image, caption="Your Selfie", width=250)

# Online URL
elif option == "🌐 Use Online Image URL":
    url = st.text_input("Enter image URL", "https://raw.githubusercontent.com/Hari99-ai/which-Bollywood-celebrity-are-you-main/main/data/ami/Aamir.44.jpg")
    if url:
        try:
            response = requests.get(url)
            user_image = Image.open(BytesIO(response.content)).convert("RGB")
            user_image_path = "user_image.jpg"
            user_image.save(user_image_path)
            st.image(user_image, caption="Online Image", width=250)
        except Exception as e:
            st.error(f"❌ Failed to load image: {e}")

# -----------------------------
# Match Button
# -----------------------------
if user_image_path and st.button("🔍 Find My Celebrity Match!"):
    st.info("🔄 Computing similarity...")
    user_embedding = DeepFace.represent(user_image_path, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
    
    best_match = None
    highest_sim = 0
    
    for celeb_name, celeb_url in CELEB_DATA.items():
        response = requests.get(celeb_url)
        celeb_img = Image.open(BytesIO(response.content)).convert("RGB")
        celeb_img.save(f"{celeb_name}.jpg")
        celeb_embedding = DeepFace.represent(f"{celeb_name}.jpg", model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
        
        sim = cosine_similarity(np.array(user_embedding).reshape(1,-1), np.array(celeb_embedding).reshape(1,-1))[0][0]*100
        if sim > highest_sim:
            highest_sim = sim
            best_match = (celeb_name, celeb_img, sim)
    
    if best_match:
        st.success(f"🎉 Best Match: {best_match[0]} ({highest_sim:.1f}% similarity)")
        st.image(best_match[1], caption=best_match[0], width=250)
    else:
        st.warning("❌ No match found.")
