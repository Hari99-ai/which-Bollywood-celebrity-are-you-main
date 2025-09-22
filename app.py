import os
import pickle
import numpy as np
from PIL import Image
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import zipfile
import requests

# Fix TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import DeepFace with error handling
try:
    from deepface import DeepFace
except ImportError as e:
    st.error(f"Failed to import DeepFace: {e}")
    st.stop()

# Configure page
st.set_page_config(
    page_title="Bollywood Celebrity Matcher",
    page_icon="🎬",
    layout="wide"
)

# Your actual Google Drive file IDs
DRIVE_FILES = {
    "embedding.pkl": "1Pv5dst2ApYrnrm-6iJPKgTflu9dKaT47",
    "successful_filenames.pkl": "14exUeyKybihWVYp2XPmcJwVWbvrvKled",
    "celebrity_db_folder": "1CJqLClJcfQH8Rd5bjnb4DHcJbkMXehh5"
}

# Direct download URLs
DRIVE_URLS = {
    "embedding.pkl": f"https://drive.google.com/uc?id={DRIVE_FILES['embedding.pkl']}",
    "successful_filenames.pkl": f"https://drive.google.com/uc?id={DRIVE_FILES['successful_filenames.pkl']}"
}

def download_file_from_gdrive(file_id, output_path, file_name):
    """Download file from Google Drive using gdown"""
    try:
        st.info(f"📥 Downloading {file_name}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        st.success(f"✅ Downloaded {file_name}")
        return True
    except Exception as e:
        st.error(f"❌ Failed to download {file_name}: {e}")
        return False

def download_celebrity_folder():
    """Download celebrity database folder from Google Drive"""
    celebrity_folder_path = "celebrity_db"
    
    if os.path.exists(celebrity_folder_path) and os.listdir(celebrity_folder_path):
        return True
    
    try:
        st.info("📥 Downloading celebrity database folder...")
        
        # Create celebrity_db directory
        os.makedirs(celebrity_folder_path, exist_ok=True)
        
        # Download the entire folder as zip
        folder_url = f"https://drive.google.com/uc?id={DRIVE_FILES['celebrity_db_folder']}&export=download"
        
        # Try to download folder contents
        # Note: This might require the folder to be zipped on Google Drive
        try:
            gdown.download_folder(
                f"https://drive.google.com/drive/folders/{DRIVE_FILES['celebrity_db_folder']}",
                output=celebrity_folder_path,
                quiet=False
            )
            st.success("✅ Downloaded celebrity database")
            return True
        except:
            # Fallback: If folder download fails, create a note for user
            st.warning("⚠️ Celebrity images folder download failed. App will work but won't display celebrity images.")
            return False
            
    except Exception as e:
        st.warning(f"⚠️ Could not download celebrity folder: {e}")
        return False

def setup_files():
    """Download and setup all required files"""
    files_ready = True
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    
    # Download embedding.pkl if not exists
    if not os.path.exists("embedding.pkl"):
        if not download_file_from_gdrive(
            DRIVE_FILES["embedding.pkl"], 
            "embedding.pkl", 
            "embedding.pkl"
        ):
            files_ready = False
    
    # Download successful_filenames.pkl if not exists
    if not os.path.exists("successful_filenames.pkl"):
        if not download_file_from_gdrive(
            DRIVE_FILES["successful_filenames.pkl"], 
            "successful_filenames.pkl", 
            "successful_filenames.pkl"
        ):
            files_ready = False
    
    # Download celebrity folder
    download_celebrity_folder()
    
    return files_ready

@st.cache_data
def load_celebrity_data():
    """Load pre-computed celebrity embeddings and filenames"""
    try:
        with open("embedding.pkl", "rb") as f:
            features = pickle.load(f)
        with open("successful_filenames.pkl", "rb") as f:
            filenames = pickle.load(f)
        return features, filenames
    except Exception as e:
        st.error(f"Error loading celebrity data: {e}")
        return None, None

def extract_user_features(img_path):
    """Extract features from user uploaded image"""
    try:
        embedding = DeepFace.represent(
            img_path=img_path, 
            model_name='VGG-Face', 
            enforce_detection=False
        )
        return np.array(embedding[0]["embedding"])
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def find_best_matches(user_features, celebrity_features, filenames, top_k=3):
    """Find top K celebrity matches using cosine similarity"""
    similarities = []
    
    for i, celeb_features in enumerate(celebrity_features):
        try:
            similarity = cosine_similarity(
                user_features.reshape(1, -1), 
                celeb_features.reshape(1, -1)
            )[0][0]
            similarities.append((similarity, filenames[i]))
        except:
            continue
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:top_k]

def get_celebrity_name(filepath):
    """Extract celebrity name from file path"""
    filename = os.path.basename(filepath)
    # Remove file extension and replace underscores with spaces
    name = filename.replace("_", " ").split(".")[0]
    return name.title()

def find_local_celebrity_image(celebrity_name):
    """Try to find celebrity image in local celebrity_db folder"""
    celebrity_db_path = "celebrity_db"
    
    if not os.path.exists(celebrity_db_path):
        return None
    
    # Search for image files with similar names
    for root, dirs, files in os.walk(celebrity_db_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_name_clean = file.replace("_", " ").lower()
                celebrity_name_clean = celebrity_name.lower()
                if celebrity_name_clean in file_name_clean or any(word in file_name_clean for word in celebrity_name_clean.split()):
                    return os.path.join(root, file)
    return None

def main():
    """Main Streamlit application"""
    st.title("🎬 Which Bollywood Celebrity Are You?")
    st.markdown("Upload your photo and find your Bollywood look-alike!")
    
    # Setup files on first run
    with st.spinner("🔄 Setting up celebrity database... This may take a moment on first run."):
        files_ready = setup_files()
    
    if not files_ready:
        st.error("❌ Failed to setup required files. Please check your internet connection and try again.")
        st.stop()
    
    # Load celebrity data
    with st.spinner("📊 Loading celebrity database..."):
        celebrity_features, filenames = load_celebrity_data()
    
    if celebrity_features is None or filenames is None:
        st.error("❌ Failed to load celebrity database.")
        st.stop()
    
    st.success(f"✅ Celebrity database loaded with {len(celebrity_features)} celebrities")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clear photo of your face for best results"
        )
    
    with col2:
        # Camera input option
        camera_photo = st.camera_input("📸 Or take a selfie")
    
    # Process uploaded image or camera photo
    image_to_process = uploaded_file or camera_photo
    
    if image_to_process is not None:
        # Save uploaded image
        file_extension = "jpg" if camera_photo else image_to_process.name.split('.')[-1]
        save_path = f"uploads/user_image.{file_extension}"
        
        # Handle different image formats
            
                        st.progress(similarity_percent / 100)
                        st.markdown(f"{bar_color} **Match Quality:** {quality}")
                    
                    st.markdown("---")
                    
            else:
                st.warning("❌ No matches found. Try with a clearer, front-facing photo.")
                
        else:
            st.error("❌ Could not analyze your image. Please try with a different photo showing your face clearly.")
    
    # Instructions and tips
    with st.expander("📖 How to get the best results"):
        st.markdown("""
        **For Best Results:**
        - ✅ Use a **clear, well-lit photo** of your face
        - ✅ Make sure your **face is fully visible** and centered
        - ✅ **Remove sunglasses, masks, or hats**
        - ✅ Use **front-facing photos** (avoid side profiles)
        - ✅ Ensure **good image quality** (not blurry or pixelated)
        - ✅ **Single person** in the photo works best
        
        **Supported formats:** JPG, JPEG, PNG, WebP
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("🎬 **Bollywood Celebrity Matcher** | Built with ❤️ using Streamlit & DeepFace")

if __name__ == "__main__":
    main()            "successful_filenames.pkl"
        ):
            files_ready = False
    
    # Download celebrity folder
    download_celebrity_folder()
    
    return files_ready

@st.cache_data
def load_celebrity_data():
    """Load pre-computed celebrity embeddings and filenames"""
    try:
        with open("embedding.pkl", "rb") as f:
            features = pickle.load(f)
        with open("successful_filenames.pkl", "rb") as f:
            filenames = pickle.load(f)
        return features, filenames
    except Exception as e:
        st.error(f"Error loading celebrity data: {e}")
        return None, None

def extract_user_features(img_path):
    """Extract features from user uploaded image"""
    try:
        embedding = DeepFace.represent(
            img_path=img_path, 
            model_name='VGG-Face', 
            enforce_detection=False
        )
        return np.array(embedding[0]["embedding"])
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def find_best_matches(user_features, celebrity_features, filenames, top_k=3):
    """Find top K celebrity matches using cosine similarity"""
    similarities = []
    
    for i, celeb_features in enumerate(celebrity_features):
        try:
            similarity = cosine_similarity(
                user_features.reshape(1, -1), 
                celeb_features.reshape(1, -1)
            )[0][0]
            similarities.append((similarity, filenames[i]))
        except:
            continue
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:top_k]

def get_celebrity_name(filepath):
    """Extract celebrity name from file path"""
    filename = os.path.basename(filepath)
    # Remove file extension and replace underscores with spaces
    name = filename.replace("_", " ").split(".")[0]
    return name.title()

def find_local_celebrity_image(celebrity_name):
    """Try to find celebrity image in local celebrity_db folder"""
    celebrity_db_path = "celebrity_db"
    
    if not os.path.exists(celebrity_db_path):
        return None
    
    # Search for image files with similar names
    for root, dirs, files in os.walk(celebrity_db_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_name_clean = file.replace("_", " ").lower()
                celebrity_name_clean = celebrity_name.lower()
                if celebrity_name_clean in file_name_clean or any(word in file_name_clean for word in celebrity_name_clean.split()):
                    return os.path.join(root, file)
    return None

def main():
    """Main Streamlit application"""
    st.title("🎬 Which Bollywood Celebrity Are You?")
    st.markdown("Upload your photo and find your Bollywood look-alike!")
    
    # Setup files on first run
    with st.spinner("🔄 Setting up celebrity database... This may take a moment on first run."):
        files_ready = setup_files()
    
    if not files_ready:
        st.error("❌ Failed to setup required files. Please check your internet connection and try again.")
        st.stop()
    
    # Load celebrity data
    with st.spinner("📊 Loading celebrity database..."):
        celebrity_features, filenames = load_celebrity_data()
    
    if celebrity_features is None or filenames is None:
        st.error("❌ Failed to load celebrity database.")
        st.stop()
    
    st.success(f"✅ Celebrity database loaded with {len(celebrity_features)} celebrities")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clear photo of your face for best results"
        )
    
    with col2:
        # Camera input option
        camera_photo = st.camera_input("📸 Or take a selfie")
    
    # Process uploaded image or camera photo
    image_to_process = uploaded_file or camera_photo
    
    if image_to_process is not None:
        # Save uploaded image
        file_extension = "jpg" if camera_photo else image_to_process.name.split('.')[-1]
        save_path = f"uploads/user_image.{file_extension}"
        
        # Handle different image formats
        try:
            if file_extension.lower() == 'webp':
                img = Image.open(image_to_process)
                img = img.convert('RGB')
                save_path = "uploads/user_image.jpg"
                img.save(save_path, 'JPEG')
            else:
                with open(save_path, "wb") as f:
                    f.write(image_to_process.getbuffer())
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return
        
        # Display uploaded image
        st.image(save_path, caption="Your Photo", width=300)
        
        # Extract features from user image
        with st.spinner("🔍 Analyzing your face..."):
            user_features = extract_user_features(save_path)
        
        if user_features is not None:
            # Find celebrity matches
            with st.spinner("⭐ Finding your celebrity matches..."):
                matches = find_best_matches(user_features, celebrity_features, filenames)
            
            if matches:
                st.markdown("## 🎭 Your Celebrity Matches")
                
                # Check if top match is very high
                top_similarity = matches[0][0] * 100
                if top_similarity >= 80:
                    st.balloons()
                    st.success(f"🎉 Wow! You have {top_similarity:.1f}% similarity with your top match!")
                
                # Display matches
                for i, (similarity, celebrity_path) in enumerate(matches):
                    similarity_percent = similarity * 100
                    celebrity_name = get_celebrity_name(celebrity_path)
                    
                    # Create match display
                    st.markdown(f"### 🏆 #{i+1} {celebrity_name}")
                    
                    # Create columns for image and details
                    match_col1, match_col2 = st.columns([1, 2])
                    
                    with match_col1:
                        # Try to display celebrity image
                        celebrity_img_path = find_local_celebrity_image(celebrity_name)
                        if celebrity_img_path and os.path.exists(celebrity_img_path):
                            st.image(celebrity_img_path, width=200, caption=celebrity_name)
                        elif os.path.exists(celebrity_path):
                            st.image(celebrity_path, width=200, caption=celebrity_name)
                        else:
                            st.info(f"🎭 {celebrity_name}")
                            st.markdown("*Celebrity image not available*")
                    
                    with match_col2:
                        # Display similarity metrics
                        st.metric(
                            label="Similarity Score",
                            value=f"{similarity_percent:.1f}%"
                        )
                        
                        # Color-coded progress bar and quality assessment
                        if similarity_percent >= 80:
                            bar_color = "🟢"
                            quality = "Excellent Match!"
                            st.success(quality)
                        elif similarity_percent >= 65:
                            bar_color = "🟡"
                            quality = "Good Match"
                            st.info(quality)
                        else:
                            bar_color = "🔴"
                            quality = "Fair Match"
                            st.warning(quality)
                        
                        st.progress(similarity_percent / 100)
                        st.markdown(f"{bar_color} **Match Quality:** {quality}")
                    
                    st.markdown("---")
                    
            else:
                st.warning("❌ No matches found. Try with a clearer, front-facing photo.")
                
        else:
            st.error("❌ Could not analyze your image. Please try with a different photo showing your face clearly.")
    
    # Instructions and tips
    with st.expander("📖 How to get the best results"):
        st.markdown("""
        **For Best Results:**
        - ✅ Use a **clear, well-lit photo** of your face
        - ✅ Make sure your **face is fully visible** and centered
        - ✅ **Remove sunglasses, masks, or hats**
        - ✅ Use **front-facing photos** (avoid side profiles)
        - ✅ Ensure **good image quality** (not blurry or pixelated)
        - ✅ **Single person** in the photo works best
        
        **Supported formats:** JPG, JPEG, PNG, WebP
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("🎬 **Bollywood Celebrity Matcher** | Built with ❤️ using Streamlit & DeepFace")

if __name__ == "__main__":
    main()

