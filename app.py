import os
import pickle
import numpy as np
from PIL import Image
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import DeepFace safely with error handling
@st.cache_resource
def import_deepface():
    """Safely import DeepFace with caching"""
    try:
        from deepface import DeepFace
        return DeepFace
    except ImportError as e:
        st.error(f"❌ Failed to import DeepFace: {e}")
        st.info("Please ensure all dependencies are installed correctly.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error initializing DeepFace: {e}")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Bollywood Celebrity Matcher",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 50%, #FFD23F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .match-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #FF6B35;
    }
    
    .stProgress .st-bo {
        background-color: #FF6B35;
    }
</style>
""", unsafe_allow_html=True)

# Google Drive file IDs - Update these with your actual file IDs
DRIVE_FILES = {
    "embedding.pkl": "1Pv5dst2ApYrnrm-6iJPKgTflu9dKaT47",
    "successful_filenames.pkl": "14exUeyKybihWVYp2XPmcJwVWbvrvKled"
}

def download_file_from_gdrive(file_id, output_path, file_name):
    """Download a single file from Google Drive with better error handling"""
    if os.path.exists(output_path):
        logger.info(f"{file_name} already exists, skipping download")
        return True
        
    try:
        with st.spinner(f"📥 Downloading {file_name}... This may take a few minutes."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
            
        if os.path.exists(output_path):
            st.success(f"✅ Successfully downloaded {file_name}")
            return True
        else:
            st.error(f"❌ Download completed but file not found: {file_name}")
            return False
            
    except Exception as e:
        st.error(f"❌ Failed to download {file_name}: {str(e)}")
        st.info("💡 Please check your internet connection and try again.")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = ["uploads", "temp"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def setup_files():
    """Download and setup all required files with progress tracking"""
    setup_directories()
    
    files_status = {}
    required_files = ["embedding.pkl", "successful_filenames.pkl"]
    
    # Check which files need to be downloaded
    files_to_download = []
    for file_name in required_files:
        if not os.path.exists(file_name):
            files_to_download.append(file_name)
        else:
            files_status[file_name] = True
    
    if not files_to_download:
        st.success("✅ All required files are already available!")
        return True
    
    st.info(f"📋 Need to download {len(files_to_download)} files: {', '.join(files_to_download)}")
    
    # Download missing files
    for file_name in files_to_download:
        if file_name in DRIVE_FILES:
            success = download_file_from_gdrive(
                DRIVE_FILES[file_name], 
                file_name, 
                file_name
            )
            files_status[file_name] = success
        else:
            st.error(f"❌ No download URL configured for {file_name}")
            files_status[file_name] = False
    
    # Check if all downloads were successful
    all_success = all(files_status.values())
    
    if all_success:
        st.success("🎉 All files downloaded successfully!")
    else:
        failed_files = [f for f, status in files_status.items() if not status]
        st.error(f"❌ Failed to download: {', '.join(failed_files)}")
        st.info("💡 The app may not work correctly without these files.")
    
    return all_success

@st.cache_data
def load_celebrity_data():
    """Load embeddings and filenames with caching"""
    try:
        if not os.path.exists("embedding.pkl") or not os.path.exists("successful_filenames.pkl"):
            st.error("❌ Required data files not found. Please run setup first.")
            return None, None
            
        with st.spinner("📊 Loading celebrity database..."):
            with open("embedding.pkl", "rb") as f:
                features = pickle.load(f)
            with open("successful_filenames.pkl", "rb") as f:
                filenames = pickle.load(f)
                
        # Validate data
        if len(features) != len(filenames):
            st.warning("⚠️ Mismatch between features and filenames count")
            min_len = min(len(features), len(filenames))
            features = features[:min_len]
            filenames = filenames[:min_len]
            
        logger.info(f"Loaded {len(features)} celebrity profiles")
        return features, filenames
        
    except Exception as e:
        st.error(f"❌ Error loading celebrity data: {str(e)}")
        return None, None

def preprocess_image(image_file, max_size=(800, 800)):
    """Preprocess uploaded image"""
    try:
        # Open image
        if hasattr(image_file, 'read'):
            img = Image.open(image_file)
        else:
            img = Image.open(image_file)
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')
        
        # Resize if too large
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        return img
        
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None

def extract_user_features(img_path, deepface_model):
    """Extract features using DeepFace with better error handling"""
    try:
        with st.spinner("🔍 Analyzing facial features..."):
            # Try different models in order of preference
            models_to_try = ['VGG-Face', 'Facenet', 'OpenFace']
            
            for model_name in models_to_try:
                try:
                    embedding = deepface_model.represent(
                        img_path=img_path,
                        model_name=model_name,
                        enforce_detection=False,
                        detector_backend='opencv'  # Use opencv as it's more stable
                    )
                    
                    if embedding and len(embedding) > 0:
                        features = np.array(embedding[0]["embedding"])
                        logger.info(f"Successfully extracted features using {model_name}")
                        return features
                        
                except Exception as model_error:
                    logger.warning(f"Model {model_name} failed: {str(model_error)}")
                    continue
            
            # If all models fail
            st.error("❌ Could not extract facial features. Please try a different image.")
            return None
            
    except Exception as e:
        st.error(f"❌ Error during feature extraction: {str(e)}")
        st.info("💡 Try uploading a clearer image with a visible face.")
        return None

def find_best_matches(user_features, celebrity_features, filenames, top_k=3):
    """Compute cosine similarity to find top matches with better error handling"""
    try:
        similarities = []
        
        with st.spinner(f"⭐ Comparing with {len(celebrity_features)} celebrities..."):
            for i, celeb_features in enumerate(celebrity_features):
                try:
                    # Ensure features are numpy arrays
                    user_feat = np.array(user_features).reshape(1, -1)
                    celeb_feat = np.array(celeb_features).reshape(1, -1)
                    
                    # Check if dimensions match
                    if user_feat.shape[1] != celeb_feat.shape[1]:
                        continue
                    
                    similarity = cosine_similarity(user_feat, celeb_feat)[0][0]
                    
                    # Convert to percentage and store
                    similarity_percent = max(0, min(100, similarity * 100))
                    similarities.append((similarity_percent, filenames[i], i))
                    
                except Exception as e:
                    logger.warning(f"Error comparing with celebrity {i}: {str(e)}")
                    continue
            
        if not similarities:
            st.error("❌ Could not find any matches. Please try a different image.")
            return []
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]
        
    except Exception as e:
        st.error(f"❌ Error finding matches: {str(e)}")
        return []

def get_celebrity_name(filepath):
    """Extract celebrity name from filepath"""
    try:
        filename = os.path.basename(filepath)
        # Remove file extension and clean up
        name = filename.split('.')[0]
        name = name.replace('_', ' ').replace('-', ' ')
        # Capitalize each word
        name = ' '.join(word.capitalize() for word in name.split())
        return name
    except:
        return "Unknown Celebrity"

def display_match_results(matches, user_image_path):
    """Display match results with better formatting"""
    if not matches:
        st.warning("❌ No matches found. Please try a clearer image.")
        return
    
    st.markdown("## 🎭 Your Bollywood Celebrity Matches")
    
    # Show user image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(user_image_path, caption="Your Photo", width=250)
    
    st.markdown("---")
    
    # Check for excellent match
    top_similarity = matches[0][0]
    if top_similarity >= 80:
        st.balloons()
        st.success(f"🎉 Incredible! You have {top_similarity:.1f}% similarity with your top match!")
    elif top_similarity >= 70:
        st.success(f"🌟 Great match! {top_similarity:.1f}% similarity with your top match!")
    
    # Display each match
    for rank, (similarity, celebrity_path, celeb_index) in enumerate(matches, 1):
        celebrity_name = get_celebrity_name(celebrity_path)
        
        # Create a card-like container
        with st.container():
            st.markdown(f'<div class="match-card">', unsafe_allow_html=True)
            
            # Header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### 🏆 #{rank} {celebrity_name}")
            with col2:
                if similarity >= 80:
                    st.markdown("🔥 **Excellent**")
                elif similarity >= 65:
                    st.markdown("⭐ **Good**")
                elif similarity >= 50:
                    st.markdown("👍 **Fair**")
                else:
                    st.markdown("🤔 **Low**")
            
            # Content
            match_col1, match_col2 = st.columns([1, 2])
            
            with match_col1:
                # Try to display celebrity image if available
                if os.path.exists(celebrity_path):
                    try:
                        st.image(celebrity_path, width=200, caption=celebrity_name)
                    except:
                        st.info(f"🎭 {celebrity_name}")
                        st.markdown("*Celebrity image not available*")
                else:
                    st.info(f"🎭 {celebrity_name}")
                    st.markdown("*Celebrity image not available*")
            
            with match_col2:
                # Similarity metrics
                st.metric(
                    label="Similarity Score", 
                    value=f"{similarity:.1f}%",
                    help="Based on facial feature analysis"
                )
                
                # Progress bar
                progress_color = (
                    "🟢" if similarity >= 80 else 
                    "🟡" if similarity >= 65 else 
                    "🟠" if similarity >= 50 else "🔴"
                )
                
                st.progress(min(similarity / 100, 1.0))
                st.markdown(f"{progress_color} **Confidence Level**")
                
                # Fun fact or encouragement
                if similarity >= 80:
                    st.success("🎬 You could be their stunt double!")
                elif similarity >= 65:
                    st.info("📸 Strong resemblance detected!")
                elif similarity >= 50:
                    st.info("👀 Some similar features!")
                else:
                    st.info("🎭 Every face is unique!")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Bollywood Celebrity Matcher</h1>', 
                unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h3>Discover which Bollywood star you resemble the most! 🌟</h3>
        <p>Upload your photo or take a selfie to find your celebrity look-alike using advanced AI face recognition.</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize DeepFace
    DeepFace = import_deepface()
    
    # Setup files
    with st.spinner("🔄 Setting up the celebrity database..."):
        files_ready = setup_files()
    
    if not files_ready:
        st.error("❌ Failed to setup required files. Please check your internet connection and try again.")
        if st.button("🔄 Retry Setup"):
            st.rerun()
        st.stop()

    # Load celebrity data
    celebrity_features, filenames = load_celebrity_data()
    
    if celebrity_features is None or filenames is None:
        st.error("❌ Failed to load celebrity database.")
        st.stop()
    
    st.success(f"✅ Celebrity database ready! {len(celebrity_features)} profiles loaded.")
    
    # Main interface
    st.markdown("## 📸 Upload Your Photo")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Take Selfie"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose your image", 
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clear photo showing your face clearly"
        )
        
        if uploaded_file:
            # Display uploaded image
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(uploaded_file, caption="Uploaded Image", width=300)
    
    with tab2:
        camera_photo = st.camera_input("📸 Take a selfie")
        
        if camera_photo:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(camera_photo, caption="Your Selfie", width=300)

    # Process the selected image
    image_to_process = uploaded_file if uploaded_file else camera_photo
    
    if image_to_process is not None:
        # Process button
        if st.button("🔍 Find My Celebrity Match!", type="primary", use_container_width=True):
            try:
                # Preprocess image
                processed_img = preprocess_image(image_to_process)
                if processed_img is None:
                    st.stop()
                
                # Save processed image
                timestamp = int(time.time())
                save_path = f"uploads/user_image_{timestamp}.jpg"
                processed_img.save(save_path, 'JPEG', quality=90)
                
                # Extract features
                user_features = extract_user_features(save_path, DeepFace)
                
                if user_features is not None:
                    # Find matches
                    matches = find_best_matches(user_features, celebrity_features, filenames)
                    
                    if matches:
                        # Display results
                        display_match_results(matches, save_path)
                        
                        # Fun statistics
                        st.markdown("## 📊 Fun Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            avg_similarity = np.mean([m[0] for m in matches])
                            st.metric("Average Match", f"{avg_similarity:.1f}%")
                        
                        with col2:
                            best_match = matches[0][0]
                            st.metric("Best Match", f"{best_match:.1f}%")
                        
                        with col3:
                            celebrities_analyzed = len(celebrity_features)
                            st.metric("Celebrities Analyzed", f"{celebrities_analyzed:,}")
                        
                        # Social sharing suggestion
                        st.markdown("---")
                        st.markdown("### 🔗 Love your results?")
                        st.info("💡 Screenshot your matches to share with friends and family!")
                        
                    else:
                        st.error("❌ No matches found. Please try with a clearer front-facing photo.")
                else:
                    st.error("❌ Could not analyze your photo. Please try a different image.")
                    
                # Clean up temporary file
                try:
                    if os.path.exists(save_path):
                        os.remove(save_path)
                except:
                    pass
                    
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")
                st.info("💡 Please try again with a different image.")

    # Tips and information
    with st.expander("📖 Tips for Best Results"):
        st.markdown("""
        ### 🎯 For Optimal Matching:
        
        **✅ Do:**
        - Use a clear, high-quality photo
        - Ensure your face is well-lit and visible
        - Face the camera directly (avoid side profiles)  
        - Remove sunglasses, masks, or face coverings
        - Use recent photos for best accuracy
        
        **❌ Avoid:**
        - Blurry or low-resolution images
        - Group photos (focus on single person)
        - Heavy filters or photo editing
        - Extreme lighting conditions
        - Photos where face is partially hidden
        
        **📱 Supported Formats:** JPG, JPEG, PNG, WebP
        """)

    # About section
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        ### 🤖 How It Works:
        
        This app uses **advanced AI face recognition technology**:
        
        1. **Face Detection**: Locates and extracts your face from the photo
        2. **Feature Extraction**: Analyzes facial features using DeepFace (VGG-Face model)
        3. **Similarity Matching**: Compares your features with our Bollywood celebrity database
        4. **Results**: Shows your top 3 matches with similarity scores
        
        ### 🛡️ Privacy & Security:
        - Your photos are processed locally and not stored permanently
        - No personal data is collected or shared
        - Images are automatically deleted after processing
        
        ### 🎬 Celebrity Database:
        - Contains facial features of popular Bollywood celebrities
        - Regularly updated with new profiles
        - Uses ethically sourced, publicly available images
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🎬 <strong>Bollywood Celebrity Matcher</strong> | Built with ❤️ using Streamlit & DeepFace</p>
        <p><em>For entertainment purposes only. Results are based on facial feature analysis.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
