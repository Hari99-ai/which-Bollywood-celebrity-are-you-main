import os
import pickle
import numpy as np
from PIL import Image
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import time
import logging
import warnings

# -----------------------------
# Environment Setup
# -----------------------------
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# -----------------------------
# Import Libraries with Error Handling
# -----------------------------
@st.cache_resource
def import_libraries():
    try:
        import gdown
        from deepface import DeepFace
        return gdown, DeepFace
    except ImportError as e:
        st.error(f"❌ Required libraries missing: {e}")
        st.info("Installing required packages... Refresh after install.")
        st.stop()

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Celebrity Matcher",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main-header { text-align: center; padding: 1rem 0; 
background: linear-gradient(90deg, #FF6B35 0%, #F7931E 50%, #FFD23F 100%);
-webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; font-weight: bold; margin-bottom: 1rem; }
.stButton > button { width: 100%; background: linear-gradient(45deg, #FF6B35, #F7931E); color: white; border: none; padding: 0.75rem; border-radius: 10px; font-weight: bold; transition: transform 0.2s; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(255,107,53,0.3); }
.match-container { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; padding: 1.5rem; margin: 1rem 0; border-left: 5px solid #FF6B35; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
.metric-card { background: white; padding: 1rem; border-radius: 10px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
.gallery-container { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin: 1rem 0; }
.gallery-item { flex: 0 1 150px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Global Variables
# -----------------------------
DRIVE_FILES = {
    "embedding.pkl": "1Pv5dst2ApYrnrm-6iJPKgTflu9dKaT47",
    "successful_filenames.pkl": "14exUeyKybihWVYp2XPmcJwVWbvrvKled"
}

# GitHub raw URL base for celebrity images
GITHUB_BASE_URL = "https://raw.githubusercontent.com/Hari99-ai/which-Bollywood-celebrity-are-you-main/main/"

# -----------------------------
# Helper Functions
# -----------------------------
def setup_directories():
    for directory in ["uploads", "temp", ".streamlit"]:
        os.makedirs(directory, exist_ok=True)

def download_from_drive(file_id, output_path, file_name, gdown_module):
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True
    for attempt in range(3):
        try:
            with st.spinner(f"📥 Downloading {file_name} (Attempt {attempt+1}/3)"):
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown_module.download(url, output_path, quiet=True)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                st.success(f"✅ {file_name} downloaded")
                return True
        except Exception:
            time.sleep(2)
    st.error(f"❌ Failed to download {file_name}")
    return False

@st.cache_data
def load_celebrity_data():
    try:
        if not os.path.exists("embedding.pkl") or not os.path.exists("successful_filenames.pkl"):
            return None, None
        with open("embedding.pkl", "rb") as f:
            features = pickle.load(f)
        with open("successful_filenames.pkl", "rb") as f:
            filenames = pickle.load(f)
        # Clean invalid entries
        valid_indices = [i for i, (feat, name) in enumerate(zip(features, filenames)) if feat is not None and len(feat)>0 and name]
        features = [features[i] for i in valid_indices]
        filenames = [filenames[i] for i in valid_indices]
        return features, filenames
    except Exception as e:
        st.error(f"Error loading celebrity data: {e}")
        return None, None

def extract_features_safe(img_path, DeepFace_module):
    models = [('VGG-Face', 'opencv'), ('Facenet', 'opencv'), ('VGG-Face', 'mtcnn'), ('OpenFace', 'opencv')]
    for model_name, detector in models:
        try:
            with st.spinner(f"🔍 Analyzing with {model_name}"):
                result = DeepFace_module.represent(img_path, model_name=model_name, enforce_detection=False, detector_backend=detector)
                if result and len(result)>0 and 'embedding' in result[0]:
                    return np.array(result[0]['embedding'])
        except Exception:
            continue
    return None

def compute_similarities(user_features, celebrity_features, filenames, top_k=3):
    if user_features is None or len(celebrity_features)==0:
        return []
    similarities = []
    user_feat = np.array(user_features).reshape(1,-1)
    with st.spinner(f"⭐ Comparing with {len(celebrity_features)} celebrities"):
        for i, celeb_feat in enumerate(celebrity_features):
            try:
                if celeb_feat is None: continue
                celeb_feat = np.array(celeb_feat).reshape(1,-1)
                if user_feat.shape[1] != celeb_feat.shape[1]: continue
                sim = cosine_similarity(user_feat, celeb_feat)[0][0]
                similarities.append((max(0,min(100,sim*100)), filenames[i]))
            except Exception:
                continue
    return sorted(similarities, key=lambda x: x[0], reverse=True)[:top_k]

def get_celebrity_name(filepath):
    """
    Extract clean celebrity name from filename/path
    """
    try:
        parts = filepath.replace("\\", "/").split("/")
        name_candidate = next((p for p in reversed(parts) if p.strip()), "")
        name_candidate = os.path.splitext(name_candidate)[0]
        name_candidate = name_candidate.split('.')[0]
        name_candidate = ''.join([c for c in name_candidate if not c.isdigit()]).strip()
        name_candidate = name_candidate.replace("_", " ").replace("-", " ")
        name_candidate = ' '.join(name_candidate.split()).lower()
        return name_candidate
    except:
        return "unknown celebrity"

def find_matching_images(matched_celebrity_path, all_filenames, max_images=6):
    """
    Find all images of the same celebrity from the dataset
    """
    try:
        # Extract celebrity name from the matched path
        matched_name = get_celebrity_name(matched_celebrity_path)
        
        # Find all images with the same celebrity name
        matching_images = []
        for filepath in all_filenames:
            if get_celebrity_name(filepath) == matched_name:
                # Convert local path to GitHub URL
                github_url = convert_to_github_url(filepath)
                if github_url:
                    matching_images.append(github_url)
                    
        return matching_images[:max_images]
    except Exception as e:
        st.error(f"Error finding matching images: {e}")
        return []

def convert_to_github_url(local_path):
    """
    Convert local file path to GitHub raw URL
    """
    try:
        # Normalize path separators
        normalized_path = local_path.replace("\\", "/")
        
        # Extract the relevant part of the path (data/...)
        if "data/" in normalized_path:
            # Get everything after and including "data/"
            relative_path = normalized_path[normalized_path.find("data/"):]
            github_url = GITHUB_BASE_URL + relative_path
            return github_url
        return None
    except Exception:
        return None

def display_celebrity_gallery(celebrity_images, celebrity_name):
    """
    Display a gallery of celebrity images
    """
    if not celebrity_images:
        return
        
    st.markdown(f"### 🖼️ More photos of {celebrity_name.title()}")
    
    # Create columns for gallery display
    cols = st.columns(min(len(celebrity_images), 4))
    
    for idx, img_url in enumerate(celebrity_images):
        try:
            with cols[idx % len(cols)]:
                st.markdown(f"""
                <div class="gallery-item">
                    <img src="{img_url}" style="width: 100%; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            continue

def display_results(matches, user_img_path, all_filenames):
    if not matches:
        st.warning("❌ No matches found.")
        return

    st.markdown("## 🎭 Your Bollywood Celebrity Matches")
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(user_img_path, caption="Your Photo", width=250)
    
    top_similarity = matches[0][0]
    if top_similarity >= 75:
        st.balloons()
        st.success(f"🎉 Amazing! {top_similarity:.1f}% similarity with top match!")
    elif top_similarity >= 60:
        st.success(f"🌟 Great match! {top_similarity:.1f}% similarity!")

    st.markdown("---")

    for rank, (sim, celeb_path) in enumerate(matches, 1):
        celeb_name = get_celebrity_name(celeb_path)
        st.markdown(f'<div class="match-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([1,2])

        with col1:
            # Try to display celebrity image from GitHub
            github_url = convert_to_github_url(celeb_path)
            if github_url:
                try:
                    st.image(github_url, caption=celeb_name.title(), width=150)
                except Exception:
                    st.markdown(f"""<div style="background: linear-gradient(45deg, #FF6B35, #F7931E);
                        color:white; padding:2rem; border-radius:10px; text-align:center; margin:1rem 0;">
                        <h3>🎬</h3><h4>{celeb_name.title()}</h4></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div style="background: linear-gradient(45deg, #FF6B35, #F7931E);
                    color:white; padding:2rem; border-radius:10px; text-align:center; margin:1rem 0;">
                    <h3>🎬</h3><h4>{celeb_name.title()}</h4></div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"### 🏆 #{rank} Match")
            st.metric("Similarity Score", f"{sim:.1f}%")
            st.progress(min(sim/100,1.0))
            if sim >= 75: st.success("🔥 Excellent Match!")
            elif sim >= 60: st.info("⭐ Very Good Match")
            elif sim >= 45: st.info("👍 Good Match")
            else: st.info("🤔 Fair Match")

        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show gallery for top match only
        if rank == 1:
            matching_images = find_matching_images(celeb_path, all_filenames, max_images=6)
            if matching_images:
                display_celebrity_gallery(matching_images, celeb_name)
        
        st.markdown("---")

# -----------------------------
# Main Application
# -----------------------------
def main():
    st.markdown('<h1 class="main-header">🎬 Celebrity Matcher</h1>', unsafe_allow_html=True)
    st.markdown("""<div style="text-align:center; margin-bottom:2rem;">
    <h3>Discover your Celebrity doppelgänger! ✨</h3>
    <p>Upload your photo and let AI find your celebrity twin.</p>
    </div>""", unsafe_allow_html=True)

    gdown, DeepFace = import_libraries()
    setup_directories()

    with st.spinner("🔄 Setting up celebrity database..."):
        files_ready = all(download_from_drive(fid, fname, fname, gdown) for fname,fid in DRIVE_FILES.items())
    if not files_ready:
        st.error("❌ Setup failed. Refresh to try again.")
        if st.button("🔄 Retry"): st.rerun()
        st.stop()

    celebrity_features, filenames = load_celebrity_data()
    if celebrity_features is None:
        st.error("❌ Failed to load celebrity database.")
        st.stop()
    st.success(f"✅ Ready! {len(celebrity_features)} profiles loaded.")

    st.markdown("## 📸 Upload Your Photo")
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Take a Selfie"])
    image_file = None

    with tab1:
        uploaded_file = st.file_uploader("Choose your image", type=["jpg","jpeg","png","webp"])
        if uploaded_file:
            image_file = uploaded_file
            st.image(uploaded_file, width=300)

    # Camera input requested **only when tab clicked**
    with tab2:
        camera_photo = st.camera_input("Take a selfie")
        if camera_photo:
            image_file = camera_photo
            st.image(camera_photo, width=300)

    if image_file and st.button("🔍 Find My Celebrity Match!"):
        try:
            timestamp = int(time.time())
            save_path = f"uploads/user_{timestamp}.jpg"
            img = Image.open(image_file).convert('RGB')
            img.save(save_path, 'JPEG', quality=85)
            user_features = extract_features_safe(save_path, DeepFace)
            if user_features is not None:
                matches = compute_similarities(user_features, celebrity_features, filenames)
                if matches: display_results(matches, save_path, filenames)
                else: st.error("❌ No suitable matches found.")
            else:
                st.error("❌ Could not analyze photo. Try a clearer image.")
            os.remove(save_path)
        except Exception as e:
            st.error(f"❌ Processing error: {e}")

    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        **For Optimal Matching:** Clear, front-facing photos, no sunglasses/masks, well-lit, recent images.
        Supported formats: JPG, JPEG, PNG, WebP
        """)

    st.markdown("---")
    st.markdown("""<div style="text-align:center; color:#666; margin-top:2rem;">
        <p>🎬 <strong>Bollywood Celebrity Matcher</strong> | created by ❤️ Hari Om</p>
        <p><em>Entertainment purposes only. Results based on facial feature similarity.</em></p>
        </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

