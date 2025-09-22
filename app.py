#!pip install streamlit deepface mtcnn opencv-python-headless Pillow scikit-learn

import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# ------------------------
# Load embeddings
# ------------------------
try:
    feature_list = pickle.load(open('embedding.pkl', 'rb'))
    filenames = pickle.load(open('successful_filenames.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading embeddings: {e}")
    st.stop()

# ------------------------
# Initialize face detector
# ------------------------
detector = MTCNN()

# ------------------------
# Helper functions
# ------------------------
def save_uploaded_image(uploaded_image):
    """Save uploaded image and convert webp to png if needed"""
    try:
        os.makedirs('uploads', exist_ok=True)
        file_ext = uploaded_image.name.split('.')[-1].lower()
        file_path = os.path.join('uploads', uploaded_image.name)

        # Save uploaded file
        with open(file_path, 'wb') as f:
            f.write(uploaded_image.getbuffer())

        # Convert webp to png
        if file_ext == 'webp':
            img = Image.open(file_path).convert("RGB")
            new_file_path = os.path.splitext(file_path)[0] + ".png"
            img.save(new_file_path, "PNG")
            os.remove(file_path)  # remove original webp
            file_path = new_file_path

        return file_path
    except Exception as e:
        st.error(f"Error saving uploaded image: {e}")
        return None

def extract_features(img_path):
    """Extract facial features using DeepFace VGG-Face"""
    try:
        img = cv2.imread(img_path)
        results = detector.detect_faces(img)
        if len(results) == 0:
            st.error("No face detected in the image.")
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
    except Exception as e:
        st.error(f"Error extracting features: {str(e)[:50]}...")
        return None

def recommend_top_n(feature_list, features, n=3):
    """Return top n matching indices and scores"""
    similarity = [cosine_similarity(features.reshape(1, -1), f.reshape(1, -1))[0][0] for f in feature_list]
    top_indices = np.argsort(similarity)[::-1][:n]
    top_scores = [similarity[i] * 100 for i in top_indices]  # percentage
    return top_indices, top_scores

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Bollywood Celebrity Matcher", layout="wide")
st.markdown("<h1 style='text-align:center;color:#FF4B4B;'>🎬 Which Bollywood Celebrity Are You?</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#333;'>Upload a photo or take a selfie to find your Bollywood twin!</p>", unsafe_allow_html=True)

# Upload or webcam input
choice = st.radio("Choose input method:", ["📂 Upload Image", "📸 Take a Selfie"])
uploaded_image = None

if choice == "📂 Upload Image":
    uploaded_image = st.file_uploader("Choose an image", type=['jpg','jpeg','png','webp'])
else:
    picture = st.camera_input("Take a selfie")
    if picture:
        uploaded_image = picture

if uploaded_image:
    file_path = save_uploaded_image(uploaded_image)
    if file_path:
        display_image = Image.open(file_path)
        st.image(display_image, caption="Your Image", width=250, use_column_width=False)

        with st.spinner("🔎 Analyzing your celebrity look-alike..."):
            features = extract_features(file_path)

        if features is not None:
            top_indices, top_scores = recommend_top_n(feature_list, features, n=3)

            st.markdown("### 🌟 Top 3 Matches")
            cols = st.columns(3)
            for i, col in enumerate(cols):
                predicted_actor = " ".join(filenames[top_indices[i]].split(os.sep)[-2].split('_'))
                score = top_scores[i]

                if i == 0 and score >= 80:
                    col.markdown(f"<h3 style='text-align:center;color:#28a745;'>🎉 Congrats! You closely resemble {predicted_actor}!</h3>", unsafe_allow_html=True)

                col.markdown(f"<h4 style='text-align:center;color:#FF5733;'>{predicted_actor}</h4>", unsafe_allow_html=True)
                col.image(filenames[top_indices[i]], use_column_width=True)

                # Colorful progress bar for score
                progress_color = "linear-gradient(90deg, #FF4B4B, #FF5733, #FFC300)"
                col.markdown(f"""
                <div style='background:#ddd; border-radius:10px; overflow:hidden; height:20px; margin-top:5px;'>
                    <div style='width:{score}%; background: {progress_color}; height:100%; text-align:center; color:white; font-weight:bold; line-height:20px;'>
                        {score:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#888;'>Developed by ❤️ Hari Om</p>", unsafe_allow_html=True)
