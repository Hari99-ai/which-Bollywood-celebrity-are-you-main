import os
import pickle
import numpy as np
from PIL import Image
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import gdown
from deepface import DeepFace

# Configure Streamlit Page
st.set_page_config(
    page_title="Bollywood Celebrity Matcher",
    page_icon="🎬",
    layout="wide"
)

# ----------------------------
# Google Drive File IDs
# ----------------------------
DRIVE_FILES = {
    "embedding.pkl": "1Pv5dst2ApYrnrm-6iJPKgTflu9dKaT47",
    "successful_filenames.pkl": "14exUeyKybihWVYp2XPmcJwVWbvrvKled",
    "celebrity_db_folder": "1CJqLClJcfQH8Rd5bjnb4DHcJbkMXehh5"
}

# ----------------------------
# Download Functions
# ----------------------------
def download_file_from_gdrive(file_id, output_path, file_name):
    """Download file from Google Drive using gdown"""
    try:
        if not os.path.exists(output_path):
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
        os.makedirs(celebrity_folder_path, exist_ok=True)
        gdown.download_folder(
            f"https://drive.google.com/drive/folders/{DRIVE_FILES['celebrity_db_folder']}",
            output=celebrity_folder_path,
            quiet=False
        )
        st.success("✅ Celebrity database ready")
        return True
    except Exception as e:
        st.warning(f"⚠️ Could not download celebrity folder: {e}")
        return False

def setup_files():
    """Ensure all required files are downloaded"""
    files_ready = True
    os.makedirs("uploads", exist_ok=True)
    
    if not download_file_from_gdrive(DRIVE_FILES["embedding.pkl"], "embedding.pkl", "embedding.pkl"):
        files_ready = False
    if not download_file_from_gdrive(DRIVE_FILES["successful_filenames.pkl"], "successful_filenames.pkl", "successful_filenames.pkl"):
        files_ready = False
    download_celebrity_folder()
    return files_ready

# ----------------------------
# Data Loading
# ----------------------------
@st.cache_data
def load_celebrity_data():
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
    similarities = []
    for i, celeb_features in enumerate(celebrity_features):
        try:
            sim = cosine_similarity(
                user_features.reshape(1, -1), 
                celeb_features.reshape(1, -1)
            )[0][0]
            similarities.append((sim, filenames[i]))
        except:
            continue
    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:top_k]

def get_celebrity_name(filepath):
    filename = os.path.basename(filepath)
    name = filename.replace("_", " ").split(".")[0]
    return name.title()

def find_local_celebrity_image(celebrity_name):
    celebrity_db_path = "celebrity_db"
    if not os.path.exists(celebrity_db_path):
        return None
    for root, _, files in os.walk(celebrity_db_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                clean_name = file.replace("_", " ").lower()
                if celebrity_name.lower() in clean_name:
                    return os.path.join(root, file)
    return None

# ----------------------------
# Main App
# ----------------------------
def main():
    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #FF4B4B; font-size: 42px;'>
            🎬 Which Bollywood Celebrity Are You?
        </h1>
        <p style='text-align: center; color: gray; font-size:18px;'>
            Upload a photo or take a selfie to find your Bollywood twin ✨
        </p>
    """, unsafe_allow_html=True)
    
    # Setup files
    with st.spinner("🔄 Setting up database..."):
        files_ready = setup_files()
    if not files_ready:
        st.error("❌ Failed to setup required files. Please check your connection.")
        st.stop()
    
    # Load celebrity data
    with st.spinner("📊 Loading celebrity embeddings..."):
        celebrity_features, filenames = load_celebrity_data()
    if celebrity_features is None:
        st.error("❌ Could not load celebrity data.")
        st.stop()
    
    st.success(f"✅ Database loaded with {len(celebrity_features)} celebrities")
    
    # Upload or Webcam
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("📂 Upload your photo", type=["jpg", "jpeg", "png", "webp"])
    with col2:
        camera_photo = st.camera_input("📸 Or take a selfie")
    
    image_to_process = uploaded_file or camera_photo
    if image_to_process:
        # Save file
        ext = "jpg" if camera_photo else image_to_process.name.split('.')[-1]
        save_path = f"uploads/user_image.{ext}"
        try:
            img = Image.open(image_to_process)
            img = img.convert('RGB')
            img.save(save_path, 'JPEG')
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return
        
        st.image(save_path, caption="🖼️ Your Uploaded Photo", width=300)
        
        # Extract features
        with st.spinner("🔍 Analyzing your face..."):
            user_features = extract_user_features(save_path)
        
        if user_features is not None:
            with st.spinner("⭐ Matching with celebrities..."):
                matches = find_best_matches(user_features, celebrity_features, filenames)
            
            if matches:
                st.markdown("## 🎭 Your Top Celebrity Matches")
                
                if matches[0][0] * 100 >= 80:
                    st.balloons()
                
                for i, (similarity, celeb_path) in enumerate(matches):
                    similarity_percent = similarity * 100
                    celebrity_name = get_celebrity_name(celeb_path)
                    
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        celeb_img = find_local_celebrity_image(celebrity_name) or celeb_path
                        if os.path.exists(celeb_img):
                            st.image(celeb_img, width=200, caption=celebrity_name)
                        else:
                            st.markdown(f"🎭 {celebrity_name}")
                            st.info("Image not available")
                    
                    with col_b:
                        st.subheader(f"🏆 #{i+1} {celebrity_name}")
                        st.metric("Similarity", f"{similarity_percent:.1f}%")
                        st.progress(similarity_percent/100)
                        if similarity_percent >= 80:
                            st.success("Excellent Match ✨")
                        elif similarity_percent >= 65:
                            st.info("Good Match 👍")
                        else:
                            st.warning("Fair Match 🙂")
                    
                    st.markdown("---")
            else:
                st.warning("⚠️ No matches found. Try a clearer, front-facing photo.")

    # Tips Expander
    with st.expander("📖 Tips for Best Results"):
        st.markdown("""
        - ✅ Use a **clear, well-lit photo**
        - ✅ Face should be **fully visible and centered**
        - ❌ Avoid sunglasses, masks, hats
        - ❌ Avoid blurry or pixelated photos
        - ✅ Upload single-person images
        """)

    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center;'>🎬 Built with ❤️ using Streamlit & DeepFace</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
