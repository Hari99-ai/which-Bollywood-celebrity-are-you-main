#!pip install streamlit deepface mtcnn opencv-python-headless Pillow scikit-learn gdown

import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import gdown
import re

# ------------------------
# CONFIG: Google Drive folder with <=45 celebrity images
# ------------------------
CELEBRITY_FOLDER_ID = "YOUR_45_IMAGES_FOLDER_ID"   # 🔥 Replace with your Google Drive folder ID

# ------------------------
# Page Configuration & CSS
# ------------------------
st.set_page_config(page_title="Bollywood Celebrity Matcher", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    .main-title {
        font-size: 3rem !important;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle { text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
    .celebrity-card {
        background: linear-gradient(145deg, #f8f9fa, #ffffff);
        border-radius: 15px; padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .celebrity-name {
        font-size: 1.4rem; font-weight: bold; text-align: center;
        margin-bottom: 15px;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .winner-badge {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: white; padding: 15px; border-radius: 15px;
        text-align: center; font-weight: bold; margin: 20px 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); }
    }
    .score-display { font-size: 1.2rem; font-weight: bold; text-align: center; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ------------------------
# Download Celebrity Images (<=45)
# ------------------------
@st.cache_data
def download_celebrity_images():
    celebrity_dir = "celebrity_images"
    os.makedirs(celebrity_dir, exist_ok=True)

    try:
        st.info("📥 Downloading celebrity dataset (max 45 images)...")
        gdown.download_folder(
            f"https://drive.google.com/drive/folders/{CELEBRITY_FOLDER_ID}",
            output=celebrity_dir,
            quiet=False
        )
        st.success("✅ Celebrity images downloaded successfully!")
        return celebrity_dir
    except Exception as e:
        st.error(f"❌ Failed to download celebrity images: {e}")
        return None

# ------------------------
# Load embeddings
# ------------------------
@st.cache_data
def load_embeddings():
    try:
        feature_list = pickle.load(open('embedding.pkl', 'rb'))
        filenames = pickle.load(open('successful_filenames.pkl', 'rb'))
        return feature_list, filenames
    except Exception as e:
        st.error(f"❌ Error loading embeddings: {e}")
        return None, None

@st.cache_resource
def load_detector():
    return MTCNN()

feature_list, filenames = load_embeddings()
if feature_list is None or filenames is None:
    st.stop()

detector = load_detector()
celebrity_dir = download_celebrity_images()

# ------------------------
# Helper functions
# ------------------------
def save_uploaded_image(uploaded_image):
    os.makedirs('uploads', exist_ok=True)
    file_ext = uploaded_image.name.split('.')[-1].lower()
    file_path = os.path.join('uploads', uploaded_image.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_image.getbuffer())
    if file_ext == 'webp':
        img = Image.open(file_path).convert("RGB")
        new_file_path = os.path.splitext(file_path)[0] + ".png"
        img.save(new_file_path, "PNG")
        os.remove(file_path)
        file_path = new_file_path
    return file_path

def extract_features(img_path):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    if len(results) == 0:
        st.error("❌ No face detected. Please try again with a clearer photo.")
        return None
    x, y, w, h = results[0]['box']
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224))
    embedding = DeepFace.represent(
        img_path=face,
        model_name='VGG-Face',
        enforce_detection=False,
        detector_backend='opencv'
    )
    if isinstance(embedding, list) and len(embedding) > 0:
        return np.array(embedding[0]['embedding'])
    elif isinstance(embedding, dict) and 'embedding' in embedding:
        return np.array(embedding['embedding'])
    return None

def recommend_top_n(feature_list, features, n=3):
    similarity = [cosine_similarity(features.reshape(1, -1), f.reshape(1, -1))[0][0] for f in feature_list]
    top_indices = np.argsort(similarity)[::-1][:n]
    top_scores = [similarity[i] * 100 for i in top_indices]
    return top_indices, top_scores

def extract_celebrity_name(file_path):
    file_path = file_path.replace('\\', '/')
    celebrity_name = os.path.basename(os.path.dirname(file_path))
    if not celebrity_name or len(celebrity_name) < 3:
        celebrity_name = os.path.splitext(os.path.basename(file_path))[0]
    celebrity_name = re.sub(r'\d+', '', celebrity_name)
    celebrity_name = celebrity_name.replace('_', ' ').replace('-', ' ')
    return ' '.join(word.capitalize() for word in celebrity_name.split() if word)

def create_progress_bar(score):
    color = "#28a745" if score >= 80 else "#ffc107" if score >= 65 else "#dc3545"
    return f"""
    <div style="background:#ddd; border-radius:25px; margin:10px 0;">
        <div style="width:{score:.1f}%; background:{color}; height:25px; border-radius:25px; text-align:center; color:white;">
            {score:.1f}%
        </div>
    </div>
    """

# ------------------------
# Main UI
# ------------------------
st.markdown('<h1 class="main-title">🎬 Which Bollywood Celebrity Are You?</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">✨ Upload a photo or take a selfie to discover your Bollywood twin! ✨</p>', unsafe_allow_html=True)

st.success(f"🎭 Celebrity database loaded! ({len(feature_list)} faces ready)")

choice = st.radio("🎯 Choose input method:", ["📂 Upload Image", "📸 Take a Selfie"], horizontal=True)
uploaded_image = None
if choice == "📂 Upload Image":
    uploaded_image = st.file_uploader("🖼️ Choose an image", type=['jpg','jpeg','png','webp'])
else:
    picture = st.camera_input("📸 Take a selfie")
    if picture: uploaded_image = picture

if uploaded_image:
    file_path = save_uploaded_image(uploaded_image)
    if file_path:
        st.image(Image.open(file_path), caption="✨ Your Photo ✨", width=300)
        with st.spinner("🔍 Finding your Bollywood twin..."):
            features = extract_features(file_path)

        if features is not None:
            top_indices, top_scores = recommend_top_n(feature_list, features, n=3)

            if top_scores[0] >= 80:
                st.balloons()
                winner = extract_celebrity_name(filenames[top_indices[0]])
                st.markdown(f'<div class="winner-badge">🎉 WOW! You look like {winner} ({top_scores[0]:.1f}% match)</div>', unsafe_allow_html=True)

            st.markdown("## 🏆 Your Top 3 Matches")
            for i in range(3):
                celeb_name = extract_celebrity_name(filenames[top_indices[i]])
                score = top_scores[i]

                st.markdown('<div class="celebrity-card">', unsafe_allow_html=True)
                st.markdown(f'<h3 class="celebrity-name">#{i+1} 🎭 {celeb_name}</h3>', unsafe_allow_html=True)
                st.markdown(f'<div class="score-display">Similarity: {score:.1f}%</div>', unsafe_allow_html=True)
                st.markdown(create_progress_bar(score), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("### 🎬 Bollywood Celebrity Matcher")
st.markdown("**Developed with ❤️ by Hari Om**")
st.markdown("*Powered by AI & Deep Learning* ✨")
