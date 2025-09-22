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

# Multiple Google Drive folder IDs (split your large folder into smaller ones)
CELEBRITY_FOLDERS = [
    "YOUR_FOLDER_ID_1",  # First 50 celebrities
    "YOUR_FOLDER_ID_2",  # Next 50 celebrities  
    "YOUR_FOLDER_ID_3",  # Next 50 celebrities
    # Add more folder IDs as needed
]

# ------------------------
# Page Configuration & CSS (same as before)
# ------------------------
st.set_page_config(page_title="Bollywood Celebrity Matcher", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    .main-title {
        font-size: 3rem !important;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-weight: bold;
        animation: gradient 3s ease infinite;
        margin-bottom: 0.5rem;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .celebrity-card {
        background: linear-gradient(145deg, #f8f9fa, #ffffff);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .celebrity-card:hover {
        transform: translateY(-5px);
    }
    
    .celebrity-name {
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .winner-badge {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: white;
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .score-display {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------
# Download from multiple folders
# ------------------------
@st.cache_data
def download_celebrity_images_batch():
    """Download from multiple Google Drive folders"""
    celebrity_dir = "celebrity_images"
    os.makedirs(celebrity_dir, exist_ok=True)
    
    success_count = 0
    
    if not CELEBRITY_FOLDERS or CELEBRITY_FOLDERS == ["YOUR_FOLDER_ID_1", "YOUR_FOLDER_ID_2", "YOUR_FOLDER_ID_3"]:
        st.warning("⚠️ Please update CELEBRITY_FOLDERS with your actual Google Drive folder IDs")
        return None
    
    for i, folder_id in enumerate(CELEBRITY_FOLDERS):
        try:
            st.info(f"📥 Downloading batch {i+1}/{len(CELEBRITY_FOLDERS)}...")
            
            batch_dir = os.path.join(celebrity_dir, f"batch_{i+1}")
            
            gdown.download_folder(
                f"https://drive.google.com/drive/folders/{folder_id}",
                output=batch_dir,
                quiet=False
            )
            
            success_count += 1
            st.success(f"✅ Downloaded batch {i+1}")
            
        except Exception as e:
            st.warning(f"⚠️ Failed to download batch {i+1}: {e}")
            continue
    
    if success_count > 0:
        st.success(f"✅ Downloaded {success_count}/{len(CELEBRITY_FOLDERS)} batches!")
        return celebrity_dir
    else:
        st.error("❌ Failed to download any celebrity images")
        return None

# ------------------------
# Load embeddings and initialize
# ------------------------
@st.cache_data
def load_embeddings():
    try:
        feature_list = pickle.load(open('embedding.pkl', 'rb'))
        filenames = pickle.load(open('successful_filenames.pkl', 'rb'))
        return feature_list, filenames
    except Exception as e:
        st.error(f"❌ Error loading celebrity database: {e}")
        return None, None

@st.cache_resource
def load_detector():
    return MTCNN()

feature_list, filenames = load_embeddings()
if feature_list is None or filenames is None:
    st.stop()

detector = load_detector()

# Download celebrity images
celebrity_dir = download_celebrity_images_batch()

# ------------------------
# Helper functions
# ------------------------
def save_uploaded_image(uploaded_image):
    """Save uploaded image and convert webp to png if needed"""
    try:
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
    except Exception as e:
        st.error(f"❌ Error saving uploaded image: {e}")
        return None

def extract_features(img_path):
    """Extract facial features using DeepFace VGG-Face"""
    try:
        img = cv2.imread(img_path)
        results = detector.detect_faces(img)
        if len(results) == 0:
            st.error("❌ No face detected in the image. Please try with a clearer photo showing your face.")
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
        st.error(f"❌ Error analyzing your photo. Please try with a different image.")
        return None

def recommend_top_n(feature_list, features, n=3):
    """Return top n matching indices and scores"""
    similarity = [cosine_similarity(features.reshape(1, -1), f.reshape(1, -1))[0][0] for f in feature_list]
    top_indices = np.argsort(similarity)[::-1][:n]
    top_scores = [similarity[i] * 100 for i in top_indices]
    return top_indices, top_scores

def extract_celebrity_name(file_path):
    """Extract celebrity name from file path"""
    try:
        file_path = file_path.replace('\\', '/')
        parts = file_path.split('/')
        
        celebrity_name = None
        
        if len(parts) >= 2:
            potential_name = parts[-2]
            if potential_name.lower() not in ['data', 'images', 'celebrity_db', 'bollywood_celeb_faces_0', 'dataset', 'main']:
                celebrity_name = potential_name
        
        if not celebrity_name or len(celebrity_name) < 3:
            filename = os.path.basename(file_path)
            celebrity_name = os.path.splitext(filename)[0]
        
        if celebrity_name:
            celebrity_name = celebrity_name.replace('_', ' ').replace('-', ' ')
            import re
            celebrity_name = re.sub(r'\d+', '', celebrity_name)
            celebrity_name = ' '.join(word.capitalize() for word in celebrity_name.split() if word)
            
            if celebrity_name.lower() in ['main', 'image', 'photo', 'pic', '']:
                return f"Bollywood Star #{hash(file_path) % 100}"
            
            return celebrity_name if celebrity_name else "Mystery Celebrity"
        
        return "Bollywood Celebrity"
    except:
        return "Unknown Celebrity"

def find_celebrity_image(original_path, celebrity_name, celebrity_dir):
    """Search for celebrity image in downloaded folders"""
    if not celebrity_dir or not os.path.exists(celebrity_dir):
        return None
    
    # Search in all batch folders
    for root, dirs, files in os.walk(celebrity_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                file_lower = file.lower().replace('_', ' ').replace('-', ' ')
                name_lower = celebrity_name.lower()
                
                # Check for name match
                if any(word in file_lower for word in name_lower.split()) or \
                   any(word in name_lower for word in file_lower.split()):
                    return os.path.join(root, file)
    
    return None

def create_progress_bar(score):
    """Create animated progress bar"""
    if score >= 80:
        gradient = "linear-gradient(90deg, #28a745, #20c997)"
    elif score >= 65:
        gradient = "linear-gradient(90deg, #ffc107, #fd7e14)"
    else:
        gradient = "linear-gradient(90deg, #dc3545, #e74c3c)"
    
    return f"""
    <div style="background: #e0e0e0; border-radius: 25px; padding: 3px; margin: 10px 0;">
        <div style="
            width: {score}%; 
            background: {gradient}; 
            height: 25px; 
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 2s ease-in-out;
        ">
            {score:.1f}%
        </div>
    </div>
    """

def display_celebrity_image(original_path, celebrity_name, rank, celebrity_dir):
    """Display celebrity image with enhanced search"""
    # Try original path first
    if os.path.exists(original_path):
        st.image(original_path, use_container_width=True, caption=f"🎭 {celebrity_name}")
        return
    
    # Search in downloaded folders
    found_image = find_celebrity_image(original_path, celebrity_name, celebrity_dir)
    if found_image and os.path.exists(found_image):
        st.image(found_image, use_container_width=True, caption=f"🎭 {celebrity_name}")
        st.success(f"✅ Found image for {celebrity_name}!")
        return
    
    # Fallback placeholder
    rank_colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
    rank_color = rank_colors[rank] if rank < 3 else "#9B59B6"
    rank_medals = ["🏆", "🥈", "🥉"]
    medal = rank_medals[rank] if rank < 3 else "🎭"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {rank_color}20, {rank_color}40);
        border: 3px solid {rank_color};
        color: #333;
        padding: 40px 20px;
        text-align: center;
        border-radius: 15px;
        font-size: 1.1rem;
        font-weight: bold;
        margin: 10px 0;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div style="font-size: 3rem; margin-bottom: 15px;">{medal}</div>
        <div style="font-size: 1.3rem; margin: 10px 0; color: {rank_color};">{celebrity_name}</div>
        <div style="font-size: 0.9rem; color: #666;">Match #{rank + 1}</div>
        <div style="font-size: 0.8rem; color: #888; margin-top: 10px;">Image not available</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------
# Main UI
# ------------------------

# Header
st.markdown('<h1 class="main-title">🎬 Which Bollywood Celebrity Are You?</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">✨ Upload a photo or take a selfie to discover your Bollywood twin! ✨</p>', unsafe_allow_html=True)

st.success(f"🎭 Celebrity database loaded successfully! ({len(feature_list)} celebrities ready for matching)")

# Upload method selection
choice = st.radio("🎯 Choose input method:", ["📂 Upload Image", "📸 Take a Selfie"], horizontal=True)
uploaded_image = None

if choice == "📂 Upload Image":
    uploaded_image = st.file_uploader(
        "🖼️ Choose your best photo", 
        type=['jpg','jpeg','png','webp'],
        help="For best results, use a clear, front-facing photo with good lighting"
    )
else:
    st.markdown("📸 **Take a selfie below:**")
    picture = st.camera_input("Smile! 😊")
    if picture:
        uploaded_image = picture

# Process uploaded image
if uploaded_image:
    file_path = save_uploaded_image(uploaded_image)
    if file_path:
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            display_image = Image.open(file_path)
            st.image(display_image, caption="✨ Your Photo ✨", width=300)

        # Analysis
        with st.spinner("🔍 Analyzing your celebrity look-alike... This might take a moment!"):
            features = extract_features(file_path)

        if features is not None:
            top_indices, top_scores = recommend_top_n(feature_list, features, n=3)

            # Special congratulations
            if top_scores[0] >= 80:
                st.balloons()
                winner_name = extract_celebrity_name(filenames[top_indices[0]])
                st.markdown(f"""
                <div class="winner-badge">
                    🎉 AMAZING! 🎉<br>
                    You're practically twins with {winner_name}!<br>
                    {top_scores[0]:.1f}% similarity - That's incredible! ✨
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("## 🏆 Your Top 3 Bollywood Celebrity Matches")
            
            # Display matches
            for i in range(3):
                celebrity_name = extract_celebrity_name(filenames[top_indices[i]])
                score = top_scores[i]
                
                st.markdown('<div class="celebrity-card">', unsafe_allow_html=True)
                
                img_col, info_col = st.columns([1, 1])
                
                with img_col:
                    display_celebrity_image(filenames[top_indices[i]], celebrity_name, i, celebrity_dir)
                
                with info_col:
                    ranking_emoji = "🏆" if i == 0 else "🥈" if i == 1 else "🥉"
                    
                    st.markdown(f'<h3 class="celebrity-name">#{i+1} {ranking_emoji} {celebrity_name}</h3>', unsafe_allow_html=True)
                    
                    score_color = "#28a745" if score >= 80 else "#ffc107" if score >= 65 else "#dc3545"
                    st.markdown(f"""
                    <div class="score-display" style="background: {score_color}20; border: 2px solid {score_color};">
                        Similarity: <span style="color: {score_color};">{score:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(create_progress_bar(score), unsafe_allow_html=True)
                    
                    if score >= 80:
                        st.success("🎯 Excellent Match!")
                    elif score >= 65:
                        st.info("🎯 Good Match!")
                    else:
                        st.warning("🎯 Fair Match")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                if i < 2:
                    st.markdown("---")

# Tips section
with st.expander("💡 Tips for Better Results"):
    st.markdown("""
    **🎯 For the most accurate matches:**
    - ✅ Use a **clear, high-quality photo**
    - ✅ Ensure your **face is well-lit** and visible
    - ✅ **Front-facing photos** work best
    - ✅ **Remove sunglasses** or face coverings
    - ✅ **Single person** in the photo
    - ✅ **Neutral expression** gives better results
    
    **📱 Supported formats:** JPG, JPEG, PNG, WebP
    """)

# Footer
st.markdown("---")
st.markdown("### 🎬 Bollywood Celebrity Matcher")
st.markdown("**Developed with ❤️ by Hari Om**")
st.markdown("*Powered by AI & Deep Learning* ✨")
