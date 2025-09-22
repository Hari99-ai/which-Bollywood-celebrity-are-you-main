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
# CONFIG: Use your shared IDs here
# ------------------------
CELEBRITY_FOLDER_ID = "1CJqLClJcfQH8Rd5bjnb4DHcJbkMXehh5"  # or the second folder if this one has ≤45 images

EMBEDDING_FILE_ID = "1Pv5dst2ApYrnrm-6iJPKgTflu9dKaT47"
FILENAMES_FILE_ID = "14exUeyKybihWVYp2XPmcJwVWbvrvKled"

# ------------------------
# Page Configuration & CSS
# ------------------------
st.set_page_config(page_title="Bollywood Celebrity Matcher", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 3rem !important; background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; font-weight: bold;
        margin-bottom: 0.5rem; }
    .subtitle { text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
    .celebrity-card { background: linear-gradient(145deg, #f8f9fa, #ffffff); border-radius: 15px; padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); margin-bottom: 20px; }
    .celebrity-name { font-size: 1.4rem; font-weight: bold; text-align: center; margin-bottom: 15px;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; }
    .winner-badge { background: linear-gradient(45deg, #FFD700, #FFA500); color: white; padding: 15px;
        border-radius: 15px; text-align: center; font-weight: bold; margin: 20px 0;
        animation: pulse 2s infinite; }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
    .score-display { font-size: 1.2rem; font-weight: bold; text-align: center; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ------------------------
# Download embeddings and filenames
# ------------------------
@st.cache_data
def download_embeddings():
    try:
        # Download embedding.pkl
        embed_url = f"https://drive.google.com/uc?id={EMBEDDING_FILE_ID}"
        if not os.path.exists("embedding.pkl"):
            st.info("📥 Downloading embedding.pkl …")
            gdown.download(embed_url, "embedding.pkl", fuzzy=True)
            st.success("✅ embedding.pkl downloaded")

        # Download successful_filenames.pkl
        fn_url = f"https://drive.google.com/uc?id={FILENAMES_FILE_ID}"
        if not os.path.exists("successful_filenames.pkl"):
            st.info("📥 Downloading filenames file …")
            gdown.download(fn_url, "successful_filenames.pkl", fuzzy=True)
            st.success("✅ filenames file downloaded")

        # Load them
        feature_list = pickle.load(open('embedding.pkl', 'rb'))
        filenames = pickle.load(open('successful_filenames.pkl', 'rb'))
        return feature_list, filenames
    except Exception as e:
        st.error(f"❌ Error downloading/loading embedding files: {e}")
        return None, None

# ------------------------
# Download celebrity images folder
# ------------------------
@st.cache_data
def download_celebrity_images():
    celebrity_dir = "celebrity_images"
    os.makedirs(celebrity_dir, exist_ok=True)

    try:
        st.info("📥 Downloading celebrity images …")
        # use folder link
        folder_url = f"https://drive.google.com/drive/folders/{CELEBRITY_FOLDER_ID}"
        gdown.download_folder(folder_url, output=celebrity_dir, quiet=False)
        st.success("✅ Celebrity images downloaded")
        return celebrity_dir
    except Exception as e:
        st.error(f"❌ Error downloading celebrity images: {e}")
        return None

# ------------------------
# Load detector
# ------------------------
@st.cache_resource
def get_detector():
    return MTCNN()

# ------------------------
# Load data
# ------------------------
feature_list, filenames = download_embeddings()
if feature_list is None or filenames is None:
    st.stop()

detector = get_detector()
celebrity_dir = download_celebrity_images()

# ------------------------
# Helper functions (same as earlier)
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
        st.error("❌ No face detected. Please try a clearer photo.")
        return None
    x, y, w, h = results[0]['box']
    x, y = max(0, x), max(0, y)
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
    sims = [cosine_similarity(features.reshape(1, -1), f.reshape(1, -1))[0][0] for f in feature_list]
    top_indices = np.argsort(sims)[::-1][:n]
    top_scores = [sims[i] * 100 for i in top_indices]
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

def find_celebrity_image(original_path, celebrity_name, celebrity_dir):
    if not celebrity_dir or not os.path.exists(celebrity_dir):
        return None
    for root, dirs, files in os.walk(celebrity_dir):
        for file in files:
            if file.lower().endswith(('.jpg','jpeg','png','webp')):
                file_lower = file.lower().replace('_',' ').replace('-',' ')
                name_lower = celebrity_name.lower()
                if any(word in file_lower for word in name_lower.split()) or any(word in name_lower for word in file_lower.split()):
                    return os.path.join(root, file)
    return None

def display_celebrity_entry(rank, celeb_name, score, original_path):
    st.markdown("<div class='celebrity-card'>", unsafe_allow_html=True)
    cols = st.columns([1,2])
    with cols[0]:
        img_path = find_celebrity_image(original_path, celeb_name, celebrity_dir)
        if img_path:
            st.image(img_path, width=200, caption=celeb_name)
        else:
            # fallback: just show name placeholder box
            st.markdown(f"**{celeb_name}**")
    with cols[1]:
        st.markdown(f"<h3 class='celebrity-name'>#{rank+1} 🎭 {celeb_name}</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='score-display'>Similarity: {score:.1f}%</div>", unsafe_allow_html=True)
        st.markdown(create_progress_bar(score), unsafe_allow_html=True)
        if score >=80:
            st.success("🎯 Excellent match!")
        elif score >=65:
            st.info("🎯 Good match")
        else:
            st.warning("🎯 Fair match")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------
# Main UI
# ------------------------
st.markdown('<h1 class="main-title">🎬 Which Bollywood Celebrity Are You?</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">✨ Upload a photo or take a selfie to discover your Bollywood twin! ✨</p>', unsafe_allow_html=True)

st.success(f"🎭 Embeddings & names loaded! ({len(feature_list)} celebrities)")

choice = st.radio("🎯 Choose input method:", ["📂 Upload Image", "📸 Take a Selfie"], horizontal=True)
uploaded_image = None
if choice == "📂 Upload Image":
    uploaded_image = st.file_uploader("🖼️ Choose an image", type=['jpg','jpeg','png','webp'])
else:
    picture = st.camera_input("📸 Take a selfie")
    if picture:
        uploaded_image = picture

if uploaded_image:
    saved_path = save_uploaded_image(uploaded_image)
    if saved_path:
        st.image(Image.open(saved_path), caption="✨ Your Photo ✨", width=300)
        with st.spinner("🔍 Matching with celebrities…"):
            user_feat = extract_features(saved_path)
        if user_feat is not None:
            top_idxs, top_scores = recommend_top_n(feature_list, user_feat, n=3)
            if top_scores[0] >= 80:
                st.balloons()
                winner = extract_celebrity_name(filenames[top_idxs[0]])
                st.markdown(f'<div class="winner-badge">🎉 Wow! You look like {winner} ({top_scores[0]:.1f}% match)</div>', unsafe_allow_html=True)

            st.markdown("## 🏆 Top 3 Celebrity Matches")
            for i in range(3):
                celeb_name = extract_celebrity_name(filenames[top_idxs[i]])
                score = top_scores[i]
                display_celebrity_entry(i, celeb_name, score, filenames[top_idxs[i]])

# Footer
st.markdown("---")
st.markdown("### 🎬 Bollywood Celebrity Matcher")
st.markdown("**Developed with ❤️ by Hari Om**")
st.markdown("*Powered by AI & Deep Learning* ✨")
