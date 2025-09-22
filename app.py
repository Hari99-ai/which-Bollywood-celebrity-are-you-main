import os
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image

st.title("🎬 Bollywood Celebrity Images from Kaggle")

# -----------------------------
# Kaggle API Credentials
# -----------------------------
os.environ['KAGGLE_USERNAME'] = 'hari9931'
os.environ['KAGGLE_KEY'] = 'ebc5af08fae31d15747b946396bb6be3'

# -----------------------------
# Initialize Kaggle API
# -----------------------------
api = KaggleApi()
api.authenticate()

# -----------------------------
# Download Dataset
# -----------------------------
dataset = 'hari9931/bollywood-celeb-images'  # your Kaggle dataset path
save_path = 'celebrity_db'
os.makedirs(save_path, exist_ok=True)

st.info("📥 Downloading dataset from Kaggle...")
api.dataset_download_files(dataset, path=save_path, unzip=True)
st.success("✅ Dataset downloaded!")

# -----------------------------
# Display Images
# -----------------------------
image_files = [f for f in os.listdir(save_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if image_files:
    st.write(f"Found {len(image_files)} images.")
    for img_file in image_files[:10]:  # show first 10 images for demo
        img_path = os.path.join(save_path, img_file)
        img = Image.open(img_path).convert("RGB")
        st.image(img, caption=img_file, width=200)
else:
    st.warning("No images found in the dataset!")
