#!pip install streamlit deepface mtcnn opencv-python-headless Pillow

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
# Page Configuration
# ------------------------
st.set_page_config(
    page_title="Bollywood Celebrity Matcher",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------
# Custom CSS Styling
# ------------------------
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3.5rem !important;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7);
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
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Celebrity card styling */
    .celebrity-card {
        background: linear-gradient(145deg, #f0f0f0, #ffffff);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        margin-bottom: 15px;
        transition: transform 0.3s ease;
    }
    
    .celebrity-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Celebrity name styling */
    .celebrity-name {
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Progress bar container */
    .progress-container {
        background: #e0e0e0;
        border-radius: 25px;
        padding: 3px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    /* Winner badge */
    .winner-badge {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Upload area styling */
    .upload-container {
        border: 3px dashed #4ECDC4;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(145deg, #f8f9fa, #ffffff);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: #FF6B6B;
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
    }
    
    /* Input method styling */
    .input-method {
        background: linear-gradient(145deg, #e3f2fd, #ffffff);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
    
    /* Match quality indicators */
    .excellent { color: #28a745; font-weight: bold; }
    .good { color: #ffc107; font-weight: bold; }
    .fair { color: #dc3545; font-weight: bold; }
    
    /* Spinning loader */
    .loader {
        text-align: center;
        padding: 2rem;
    }
    
    /* Score display */
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
# Load embeddings
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

feature_list, filenames = load_embeddings()
if feature_list is None or filenames is None:
    st.stop()

# ------------------------
# Initialize face detector
# ------------------------
@st.cache_resource
def load_detector():
    return MTCNN()

detector = load_detector()

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
        st.error(f"❌ Error analyzing your photo: Please try with a different image.")
        return None

def recommend_top_n(feature_list, features, n=3):
    """Return top n matching indices and scores"""
    similarity = [cosine_similarity(features.reshape(1, -1), f.reshape(1, -1))[0][0] for f in feature_list]
    top_indices = np.argsort(similarity)[::-1][:n]
    top_scores = [similarity[i] * 100 for i in top_indices]  # percentage
    return top_indices, top_scores

def get_quality_class(score):
    """Get CSS class based on score"""
    if score >= 80:
        return "excellent", "🏆"
    elif score >= 65:
        return "good", "🥈"
    else:
        return "fair", "🥉"

def create_progress_bar(score):
    """Create animated progress bar"""
    if score >= 80:
        gradient = "linear-gradient(90deg, #28a745, #20c997)"
    elif score >= 65:
        gradient = "linear-gradient(90deg, #ffc107, #fd7e14)"
    else:
        gradient = "linear-gradient(90deg, #dc3545, #e74c3c)"
    
    return f"""
    <div class="progress-container">
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

# ------------------------
# Main UI
# ------------------------

# Header
st.markdown('<h1 class="main-title">🎬 Which Bollywood Celebrity Are You?</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">✨ Upload a photo or take a selfie to discover your Bollywood twin! ✨</p>', unsafe_allow_html=True)

# Success message for loaded database
st.success(f"🎭 Celebrity database loaded successfully! ({len(feature_list)} celebrities ready for matching)")

# Input method selection with better styling
st.markdown('<div class="input-method">', unsafe_allow_html=True)
choice = st.radio(
    "🎯 Choose your input method:", 
    ["📁 Upload Image", "📸 Use Webcam"],
    horizontal=True
)
st.markdown('</div>', unsafe_allow_html=True)

uploaded_image = None

# Upload or Webcam input
if choice == "📁 Upload Image":
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    uploaded_image = st.file_uploader(
        "🖼️ Choose your best photo", 
        type=['jpg','jpeg','png','webp'],
        help="For best results, use a clear, front-facing photo with good lighting"
    )
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("📸 **Take a selfie below:**")
    picture = st.camera_input("Smile! 😊")
    if picture:
        uploaded_image = picture

# Process uploaded image
if uploaded_image:
    file_path = save_uploaded_image(uploaded_image)
    if file_path:
        # Display uploaded image in a nice container
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            display_image = Image.open(file_path)
            st.image(
                display_image, 
                caption="✨ Your Beautiful Photo ✨", 
                width=300,
                use_column_width=False
            )

        # Analysis with beautiful spinner
        with st.spinner("🔍 Analyzing your celebrity look-alike... This might take a moment!"):
            features = extract_features(file_path)

        if features is not None:
            top_indices, top_scores = recommend_top_n(feature_list, features, n=3)

            # Special congratulations for high match
            if top_scores[0] >= 80:
                st.balloons()
                winner_name = " ".join(filenames[top_indices[0]].split(os.sep)[-2].split('_')).title()
                st.markdown(f"""
                <div class="winner-badge">
                    🎉 AMAZING! 🎉<br>
                    You're practically twins with {winner_name}!<br>
                    {top_scores[0]:.1f}% similarity - That's incredible! ✨
                </div>
                """, unsafe_allow_html=True)

            # Results header
            st.markdown("---")
            st.markdown("## 🏆 Your Top 3 Bollywood Celebrity Matches")
            
            # Display matches in a more beautiful layout
            for i in range(3):
                predicted_actor = " ".join(filenames[top_indices[i]].split(os.sep)[-2].split('_')).title()
                score = top_scores[i]
                quality_class, emoji = get_quality_class(score)
                
                # Create celebrity card
                st.markdown('<div class="celebrity-card">', unsafe_allow_html=True)
                
                # Create columns for better layout
                img_col, info_col = st.columns([1, 1])
                
                with img_col:
                    try:
                        st.image(
                            filenames[top_indices[i]], 
                            use_column_width=True,
                            caption=f"🎭 {predicted_actor}"
                        )
                    except:
                        st.info(f"🎭 {predicted_actor}\n(Image not available)")
                
                with info_col:
                    st.markdown(f'<h3 class="celebrity-name">#{i+1} {emoji} {predicted_actor}</h3>', unsafe_allow_html=True)
                    
                    # Score display with styling
                    score_color = "#28a745" if score >= 80 else "#ffc107" if score >= 65 else "#dc3545"
                    st.markdown(f"""
                    <div class="score-display" style="background: {score_color}20; border: 2px solid {score_color};">
                        Similarity Score: <span style="color: {score_color};">{score:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Animated progress bar
                    st.markdown(create_progress_bar(score), unsafe_allow_html=True)
                    
                    # Quality indicator
                    quality_text = "Excellent Match!" if score >= 80 else "Good Match!" if score >= 65 else "Fair Match"
                    st.markdown(f'<p class="{quality_class}">🎯 {quality_text}</p>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                if i < 2:  # Add separator except for last item
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
st.markdown("""
<div class="footer">
    <p>🎬 <strong>Bollywood Celebrity Matcher</strong></p>
    <p>Developed with ❤️ by <strong>Hari Om</strong></p>
    <p>Powered by AI & Deep Learning ✨</p>
</div>
""", unsafe_allow_html=True)#!pip install streamlit deepface mtcnn opencv-python-headless Pillow scikit-learn

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
# Page Configuration
# ------------------------
st.set_page_config(
    page_title="Bollywood Celebrity Matcher",
    page_icon="🎬",
    layout="wide"
)

# ------------------------
# Custom CSS
# ------------------------
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
    
    .placeholder-image {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 60px 20px;
        text-align: center;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

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
        st.error(f"❌ Error loading celebrity database: {e}")
        return None, None

feature_list, filenames = load_embeddings()
if feature_list is None or filenames is None:
    st.stop()

# ------------------------
# Initialize face detector
# ------------------------
@st.cache_resource
def load_detector():
    return MTCNN()

detector = load_detector()

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
    top_scores = [similarity[i] * 100 for i in top_indices]  # percentage
    return top_indices, top_scores

def extract_celebrity_name(file_path):
    """Extract celebrity name from file path"""
    try:
        # Convert forward slashes to backslashes for Windows compatibility
        file_path = file_path.replace('\\', '/')
        
        # Split the path and get meaningful parts
        parts = file_path.split('/')
        
        # Try different approaches to extract celebrity name
        celebrity_name = None
        
        # Method 1: Look for directory name (most common)
        if len(parts) >= 2:
            potential_name = parts[-2]  # Parent directory
            if potential_name and potential_name.lower() not in ['data', 'images', 'celebrity_db', 'bollywood_celeb_faces_0', 'dataset']:
                celebrity_name = potential_name
        
        # Method 2: If no good directory name, use filename
        if not celebrity_name or len(celebrity_name) < 2:
            filename = os.path.basename(file_path)
            celebrity_name = os.path.splitext(filename)[0]
        
        # Method 3: If still no good name, try to extract from full path
        if not celebrity_name or celebrity_name.lower() in ['main', 'image', 'photo', 'pic']:
            # Look for recognizable names in the path
            path_lower = file_path.lower()
            
            # Common Bollywood celebrity names to look for
            celebrity_keywords = [
                'shah_rukh_khan', 'shahrukh_khan', 'srk',
                'salman_khan', 'aamir_khan', 'akshay_kumar',
                'hrithik_roshan', 'ranbir_kapoor', 'ranveer_singh',
                'deepika_padukone', 'priyanka_chopra', 'katrina_kaif',
                'alia_bhatt', 'kareena_kapoor', 'sonam_kapoor',
                'anushka_sharma', 'vidya_balan', 'kangana_ranaut'
            ]
            
            for keyword in celebrity_keywords:
                if keyword in path_lower:
                    celebrity_name = keyword
                    break
        
        # Clean up the name
        if celebrity_name:
            celebrity_name = celebrity_name.replace('_', ' ').replace('-', ' ')
            celebrity_name = ' '.join(word.capitalize() for word in celebrity_name.split())
            return celebrity_name
        else:
            # Fallback: use a generic name with index
            return f"Bollywood Celebrity"
            
    except Exception as e:
        return "Unknown Celebrity"

def create_progress_bar(score):
    """Create animated progress bar"""
    if score >= 80:
        gradient = "linear-gradient(90deg, #28a745, #20c997)"
        color = "#28a745"
    elif score >= 65:
        gradient = "linear-gradient(90deg, #ffc107, #fd7e14)"
        color = "#ffc107"
    else:
        gradient = "linear-gradient(90deg, #dc3545, #e74c3c)"
        color = "#dc3545"
    
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

def display_celebrity_image(file_path, celebrity_name):
    """Display celebrity image with fallback"""
    try:
        if os.path.exists(file_path):
            st.image(file_path, use_container_width=True, caption=f"🎭 {celebrity_name}")
        else:
            # Fallback: Show placeholder with celebrity name
            st.markdown(f"""
            <div class="placeholder-image">
                🎭<br>
                {celebrity_name}<br>
                <small>Image not available</small>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        # Another fallback for any image loading errors
        st.markdown(f"""
        <div class="placeholder-image">
            🎭<br>
            {celebrity_name}<br>
            <small>Image unavailable</small>
        </div>
        """, unsafe_allow_html=True)

# ------------------------
# Main UI
# ------------------------

# Header
st.markdown('<h1 class="main-title">🎬 Which Bollywood Celebrity Are You?</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">✨ Upload a photo or take a selfie to discover your Bollywood twin! ✨</p>', unsafe_allow_html=True)

# Database status
st.success(f"🎭 Celebrity database loaded successfully! ({len(feature_list)} celebrities ready for matching)")

# Upload or webcam input
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

            # Special congratulations for high match
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
                
                # Create celebrity card
                st.markdown('<div class="celebrity-card">', unsafe_allow_html=True)
                
                # Create columns for layout
                img_col, info_col = st.columns([1, 1])
                
                with img_col:
                    display_celebrity_image(filenames[top_indices[i]], celebrity_name)
                
                with info_col:
                    # Ranking emoji
                    ranking_emoji = "🏆" if i == 0 else "🥈" if i == 1 else "🥉"
                    
                    st.markdown(f'<h3 class="celebrity-name">#{i+1} {ranking_emoji} {celebrity_name}</h3>', unsafe_allow_html=True)
                    
                    # Score display
                    score_color = "#28a745" if score >= 80 else "#ffc107" if score >= 65 else "#dc3545"
                    st.markdown(f"""
                    <div class="score-display" style="background: {score_color}20; border: 2px solid {score_color};">
                        Similarity: <span style="color: {score_color};">{score:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar
                    st.markdown(create_progress_bar(score), unsafe_allow_html=True)
                    
                    # Quality indicator
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

