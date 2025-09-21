import os
import pickle
import numpy as np
from tqdm import tqdm
from deepface import DeepFace
from PIL import Image
import streamlit as st
import gdown

# -----------------------------
# Step 0: Create folders
# -----------------------------
os.makedirs("uploads", exist_ok=True)
os.makedirs("celebrity_db", exist_ok=True)

# -----------------------------
# Step 1: Download celebrity images manually from Drive folder
# -----------------------------
# Folder URL: https://drive.google.com/drive/folders/1CJqLClJcfQH8Rd5bjnb4DHcJbkMXehh5?usp=sharing
# Manually download the images and place them in "celebrity_db"
# Or, if you know individual file IDs, use this map:
celebrity_files = {
    # "shahrukh_khan.jpg": "FILE_ID",
    # "salman_khan.jpg": "FILE_ID",
}

for file_name, file_id in celebrity_files.items():
    file_path = os.path.join("celebrity_db", file_name)
    if not os.path.exists(file_path):
        try:
            gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
        except Exception as e:
            st.warning(f"Could not download {file_name}: {e}")

# -----------------------------
# Step 2: Load filenames
# -----------------------------
filenames_pkl = "successful_filenames.pkl"

if not os.path.exists(filenames_pkl):
    filenames = [os.path.join("celebrity_db", file) for file in os.listdir("celebrity_db")]
    with open(filenames_pkl, "wb") as f:
        pickle.dump(filenames, f)
else:
    with open(filenames_pkl, "rb") as f:
        filenames = pickle.load(f)

# -----------------------------
# Step 3: Extract features (if not done already)
# -----------------------------
embedding_pkl = "embedding.pkl"
if not os.path.exists(embedding_pkl):
    def feature_extractor(img_path):
        embedding = DeepFace.represent(img_path=img_path, model_name='VGG-Face', enforce_detection=False)
        return np.array(embedding[0]["embedding"])

    features = [feature_extractor(file) for file in tqdm(filenames)]
    with open(embedding_pkl, "wb") as f:
        pickle.dump(features, f)
else:
    with open(embedding_pkl, "rb") as f:
        features = pickle.load(f)

st.success("✅ Celebrity features loaded.")

# -----------------------------
# Step 4: Upload your image
# -----------------------------
st.title("Which Bollywood Celebrity Are You?")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    save_path = f"uploads/{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(save_path, caption="Your uploaded image", use_column_width=True)

    # -----------------------------
    # Step 5: Find the best celebrity match
    # -----------------------------
    try:
        result = DeepFace.find(
            img_path=save_path,
            db_path="celebrity_db",
            model_name="VGG-Face",
            enforce_detection=False
        )

        if len(result) > 0 and not result[0].empty:
            best_match = result[0].iloc[0].identity
            predicted_actor = os.path.basename(best_match).replace("_", " ").split(".")[0]
            st.success(f"✅ You look like: {predicted_actor}")

            col1, col2 = st.columns(2)
            with col1:
                st.image(save_path, caption="Your Uploaded Image")
            with col2:
                st.image(best_match, caption=f"Seems like {predicted_actor}")
        else:
            st.warning("❌ No match found. Try another photo.")

    except Exception as e:
        st.error(f"⚠️ Error during celebrity match: {e}")
