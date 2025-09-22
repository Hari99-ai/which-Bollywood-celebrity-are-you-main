import os
import pickle
import numpy as np
from PIL import Image
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import time
import logging
import warnings

# Suppress all warnings and TensorFlow logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Import libraries safely
@st.cache_resource
def import_libraries():
    """Import required libraries with error handling"""
    try:
        import gdown
        from deepface import DeepFace
        return gdown, DeepFace
    except ImportError as e:
        st.error(f"❌ Failed to import required libraries: {e}")
        st.info("Installing required packages... Please refresh the page in a moment.")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Bollywood Celebrity Matcher",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 50%, #FFD23F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(45deg, #FF6B35, #F7931E);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 107, 53, 0.3);
    }
    
    .match-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #FF6B35;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Configuration
DRIVE_FILES = {
    "embedding.pkl": "1Pv5dst2ApYrnrm-6iJPKgTflu9dKaT47",
    "successful_filenames.pkl": "14exUeyKybihWVYp2XPmcJwVWbvrvKled"
}

def setup_directories():
    """Create necessary directories"""
    for directory in ["uploads", "temp", ".streamlit"]:
        os.makedirs(directory, exist_ok=True)

def download_from_drive(file_id, output_path, file_name, gdown_module):
    """Download file from Google Drive with retry logic"""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True
        
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with st.spinner(f"📥 Downloading {file_name}... (Attempt {attempt + 1}/{max_retries})"):
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown_module.download(url, output_path, quiet=True)
                
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                st.success(f"✅ Downloaded {file_name}")
                return True
                
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"❌ Failed to download {file_name} after {max_retries} attempts")
                return False
            time.sleep(2)  # Wait before retry
    
    return False

@st.cache_data
def load_celebrity_data():
    """Load celebrity data with caching"""
    try:
        if not os.path.exists("embedding.pkl") or not os.path.exists("successful_filenames.pkl"):
            return None, None
            
        with open("embedding.pkl", "rb") as f:
            features = pickle.load(f)
        with open("successful_filenames.pkl", "rb") as f:
            filenames = pickle.load(f)
            
        # Validate and clean data
        valid_indices = []
        for i, (feat, name) in enumerate(zip(features, filenames)):
            if feat is not None and len(feat) > 0 and name:
                valid_indices.append(i)
        
        features = [features[i] for i in valid_indices]
        filenames = [filenames[i] for i in valid_indices]
        
        return features, filenames
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def extract_features_safe(img_path, deepface_module):
    """Extract features with multiple fallback options"""
    models_to_try = [
        ('VGG-Face', 'opencv'),
        ('Facenet', 'opencv'),
        ('VGG-Face', 'mtcnn'),
        ('OpenFace', 'opencv')
    ]
    
    for model_name, detector in models_to_try:
        try:
            with st.spinner(f"🔍 Analyzing with {model_name}..."):
                result = deepface_module.represent(
                    img_path=img_path,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend=detector
                )
                
                if result and len(result) > 0 and 'embedding' in result[0]:
                    return np.array(result[0]['embedding'])
                    
        except Exception:
            continue
    
    return None

def compute_similarities(user_features, celebrity_features, filenames, top_k=3):
    """Compute similarities with error handling"""
    if user_features is None or len(celebrity_features) == 0:
        return []
    
    similarities = []
    user_feat = np.array(user_features).reshape(1, -1)
    
    with st.spinner(f"⭐ Comparing with {len(celebrity_features)} celebrities..."):
        for i, celeb_features in enumerate(celebrity_features):
            try:
                if celeb_features is None:
                    continue
                    
                celeb_feat = np.array(celeb_features).reshape(1, -1)
                
                if user_feat.shape[1] != celeb_feat.shape[1]:
                    continue
                
                similarity = cosine_similarity(user_feat, celeb_feat)[0][0]
                similarity_percent = max(0, min(100, similarity * 100))
                similarities.append((similarity_percent, filenames[i]))
                
            except Exception:
                continue
    
    return sorted(similarities, key=lambda x: x[0], reverse=True)[:top_k]

def get_celebrity_name(filepath):
    """Extract clean celebrity name"""
    try:
        name = os.path.basename(filepath).split('.')[0]
        name = name.replace('_', ' ').replace('-', ' ')
        return ' '.join(word.capitalize() for word in name.split())
    except:
        return "Unknown Celebrity"

def display_results(matches, user_img_path):
    """Display results with enhanced UI"""
    if not matches:
        st.warning("❌ No matches found. Try a clearer front-facing photo.")
        return
    
    st.markdown("## 🎭 Your Bollywood Celebrity Matches")
    
    # Show user image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(user_img_path, caption="Your Photo", width=250)
    
    # Top match celebration
    top_similarity = matches[0][0]
    if top_similarity >= 75:
        st.balloons()
        st.success(f"🎉 Amazing! {top_similarity:.1f}% similarity with your top match!")
    elif top_similarity >= 60:
        st.success(f"🌟 Great match! {top_similarity:.1f}% similarity!")
    
    st.markdown("---")
    
    # Display matches
    for rank, (similarity, celebrity_path) in enumerate(matches, 1):
        celebrity_name = get_celebrity_name(celebrity_path)
        
        st.markdown(f'<div class="match-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Placeholder for celebrity image
            st.markdown(f"""
            <div style="
                background: linear-gradient(45deg, #FF6B35, #F7931E);
                color: white;
                padding: 2rem;
                border-radius: 10px;
                text-align: center;
                margin: 1rem 0;
            ">
                <h3>🎬</h3>
                <h4>{celebrity_name}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"### 🏆 #{rank} Match")
            
            # Similarity score
            st.metric("Similarity Score", f"{similarity:.1f}%")
            
            # Progress bar with custom styling
            progress_val = min(similarity / 100, 1.0)
            st.progress(progress_val)
            
            # Quality indicator
            if similarity >= 75:
                st.success("🔥 Excellent Match!")
            elif similarity >= 60:
                st.info("⭐ Very Good Match")
            elif similarity >= 45:
                st.info("👍 Good Match")
            else:
                st.info("🤔 Fair Match")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">🎬 Bollywood Celebrity Matcher</h1>', 
                unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h3>Discover your Bollywood doppelgänger! ✨</h3>
        <p>Upload your photo and let AI find your celebrity twin using advanced facial recognition.</p>
    </div>
    """, unsafe_allow_html=True)

    # Import libraries
    gdown, DeepFace = import_libraries()
    
    # Setup
    setup_directories()
    
    # Download required files
    with st.spinner("🔄 Setting up celebrity database..."):
        files_ready = True
        for file_name, file_id in DRIVE_FILES.items():
            if not download_from_drive(file_id, file_name, file_name, gdown):
                files_ready = False
    
    if not files_ready:
        st.error("❌ Setup failed. Please refresh the page to try again.")
        if st.button("🔄 Retry"):
            st.rerun()
        st.stop()

    # Load data
    celebrity_features, filenames = load_celebrity_data()
    if celebrity_features is None:
        st.error("❌ Failed to load celebrity database.")
        st.stop()
    
    st.success(f"✅ Ready! {len(celebrity_features)} celebrity profiles loaded.")
    
    # Main interface
    st.markdown("## 📸 Upload Your Photo")
    
    # Tabs for input methods
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera"])
    
    image_file = None
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose your image", 
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clear photo of your face"
        )
        if uploaded_file:
            image_file = uploaded_file
            st.image(uploaded_file, caption="Your Photo", width=300)
    
    with tab2:
        camera_photo = st.camera_input("Take a selfie")
        if camera_photo:
            image_file = camera_photo
            st.image(camera_photo, caption="Your Selfie", width=300)

    # Process image
    if image_file is not None:
        if st.button("🔍 Find My Celebrity Match!", type="primary"):
            try:
                # Save image
                timestamp = int(time.time())
                
                if hasattr(image_file, 'name') and image_file.name.lower().endswith('.webp'):
                    # Handle WebP conversion
                    img = Image.open(image_file).convert('RGB')
                    save_path = f"uploads/user_{timestamp}.jpg"
                    img.save(save_path, 'JPEG', quality=85)
                else:
                    # Save other formats
                    save_path = f"uploads/user_{timestamp}.jpg"
                    if hasattr(image_file, 'name'):
                        with open(save_path, "wb") as f:
                            f.write(image_file.getbuffer())
                    else:
                        Image.open(image_file).save(save_path)
                
                # Extract features
                user_features = extract_features_safe(save_path, DeepFace)
                
                if user_features is not None:
                    # Find matches
                    matches = compute_similarities(user_features, celebrity_features, filenames)
                    
                    if matches:
                        display_results(matches, save_path)
                        
                        # Statistics
                        st.markdown("## 📊 Analysis Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Best Match", f"{matches[0][0]:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            avg_score = np.mean([m[0] for m in matches])
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Average Score", f"{avg_score:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Celebrities Analyzed", f"{len(celebrity_features):,}")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("❌ No suitable matches found. Please try a clearer image.")
                else:
                    st.error("❌ Could not analyze your photo. Please try a different image with a clear face.")
                
                # Cleanup
                try:
                    if os.path.exists(save_path):
                        os.remove(save_path)
                except:
                    pass
                    
            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")

    # Tips section
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        **For Optimal Matching:**
        - ✅ Use clear, well-lit photos
        - ✅ Face the camera directly
        - ✅ Remove sunglasses or masks
        - ✅ Ensure face is clearly visible
        - ✅ Use recent, unfiltered photos
        
        **Supported formats:** JPG, JPEG, PNG, WebP
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🎬 <strong>Bollywood Celebrity Matcher</strong> | Built with ❤️ using Streamlit & AI</p>
        <p><em>For entertainment purposes only. Results based on facial feature similarity.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
