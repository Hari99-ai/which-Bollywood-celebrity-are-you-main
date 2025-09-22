import os
import pickle
import numpy as np
from PIL import Image
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO
import time
import warnings

# -----------------------------
# Setup
# -----------------------------
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
st.set_page_config(page_title="Bollywood Celebrity Matcher", page_icon="🎬", layout="wide")

# -----------------------------
# Import DeepFace safely
# -----------------------------
@st.cache_resource
def import_deepface():
    try:
        from deepface import DeepFace
        return DeepFace
    except ImportError:
        st.error("❌ DeepFace library not found. Install via `pip install deepface`")
        st.stop()

DeepFace = import_deepface()

# -----------------------------
# Load celebrity embeddings
# -----------------------------
@st.cache_data
def load_celebrity_data():
    try:
        with open("embedding.pkl", "rb") as f:
            features = pickle.load(f)
        with open("successful_filenames.pkl", "rb") as f:
            filenames = pickle.load(f)
        valid_indices = [i for i, (feat, name) in enumerate(zip(features, filenames)) if feat is not None and len(feat)>0 and name]
        features = [features[i] for i in valid_indices]
        filenames = [filenames[i] for i in valid_indices]
        return features, filenames
    except Exception as e:
        st.error(f"❌ Failed to load celebrity embeddings: {e}")
        return [], []

celebrity_features, celebrity_filenames = load_celebrity_data()

# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(image_path):
    try:
        result = DeepFace.represent(image_path, model_name="VGG-Face", enforce_detection=False)
        if result and 'embedding' in result[0]:
            return np.array(result[0]['embedding'])
    except Exception as e:
        st.warning(f"⚠️ Feature extraction failed: {e}")
    return None

# -----------------------------
# Compute similarity
# -----------------------------
def find_matches(user_features, celeb_features, filenames, top_k=3):
    user_feat = np.array(user_features).reshape(1, -1)
    sims = []
    for i, feat in enumerate(celeb_features):
        feat = np.array(feat).reshape(1, -1)
        if user_feat.shape[1] != feat.shape[1]:
            continue
        sim = cosine_similarity(user_feat, feat)[0][0]
        sims.append((sim*100, filenames[i]))
    sims = sorted(sims, key=lambda x: x[0], reverse=True)
    return sims[:top_k]

# -----------------------------
# Streamlit UI
# -----------------------------
st.markdown("<h1 style='text-align:center'>🎬 Bollywood Celebrity Matcher</h1>", unsafe_allow_html=True)

option = st.radio("Choose your input:", ["📁 Upload Image", "📷 Take a Selfie", "🌐 Use Online Image URL"])
user_image_path = None

# Upload local image
if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Choose your image", type=["jpg","jpeg","png","webp"])
    if uploaded_file:
        user_image = Image.open(uploaded_file).convert("RGB")
        user_image_path = f"temp_user.jpg"
        user_image.save(user_image_path)
        st.image(user_image, caption="Your Image", width=250)

# Take a selfie
elif option == "📷 Take a Selfie":
    camera_file = st.camera_input("Take a selfie")
    if camera_file:
        user_image = Image.open(camera_file).convert("RGB")
        user_image_path = f"temp_user.jpg"
        user_image.save(user_image_path)
        st.image(user_image, caption="Your Selfie", width=250)

# Online image URL
elif option == "🌐 Use Online Image URL":
    url = st.text_input("Enter image URL", "https://raw.githubusercontent.com/Hari99-ai/which-Bollywood-celebrity-are-you-main/main/data/ami/Aamir.44.jpg")
    if url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            user_image = Image.open(BytesIO(response.content)).convert("RGB")
            user_image_path = f"temp_user.jpg"
            user_image.save(user_image_path)
            st.image(user_image, caption="Online Image", width=250)
            st.success("✅ Image loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load image from URL: {e}")

# -----------------------------
# Match button
# -----------------------------
if user_image_path and st.button("🔍 Find My Celebrity Match!"):
    with st.spinner("🔄 Extracting features and computing similarity..."):
        user_features = extract_features(user_image_path)
        if user_features is not None and celebrity_features:
            matches = find_matches(user_features, celebrity_features, celebrity_filenames)
            st.markdown("## 🎭 Top Matches")
            for rank, (sim, celeb_file) in enumerate(matches,1):
                celeb_name = os.path.splitext(os.path.basename(celeb_file))[0].replace("_"," ").title()
                st.metric(f"#{rank} Match: {celeb_name}", f"{sim:.1f}% similarity")
        else:
            st.error("❌ Could not compute matches. Make sure celebrity embeddings are loaded correctly.")
