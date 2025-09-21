import os
import pickle
import numpy as np
from tqdm import tqdm
from deepface import DeepFace
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import gdown

# -----------------------------
# Step 0: Download pickle files from Google Drive
# -----------------------------
# Replace with your actual file IDs from the Drive folder
EMBEDDING_FILE_ID = "YOUR_EMBEDDING_FILE_ID"
FILENAMES_FILE_ID = "YOUR_FILENAMES_FILE_ID"

def download_if_missing(file_id, local_path):
    if not os.path.exists(local_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {local_path} ...")
        gdown.download(url, local_path, quiet=False)
    else:
        print(f"{local_path} already exists.")

download_if_missing(EMBEDDING_FILE_ID, "embedding.pkl")
download_if_missing(FILENAMES_FILE_ID, "filenames.pkl")

# -----------------------------
# Step 1: Load image filenames
# -----------------------------
filenames_pkl = "filenames.pkl"
with open(filenames_pkl, "rb") as f:
    filenames = pickle.load(f)

# -----------------------------
# Step 2: Extract features (if not done already)
# -----------------------------
embedding_pkl = "embedding.pkl"
if not os.path.exists(embedding_pkl):
    def feature_extractor(img_path):
        """Extract deep features using VGG-Face"""
        embedding = DeepFace.represent(img_path=img_path, model_name='VGG-Face', enforce_detection=False)
        return np.array(embedding[0]["embedding"])

    features = [feature_extractor(file) for file in tqdm(filenames)]
    with open(embedding_pkl, "wb") as f:
        pickle.dump(features, f)
else:
    with open(embedding_pkl, "rb") as f:
        features = pickle.load(f)

print("✅ Feature extraction completed and loaded.")

# -----------------------------
# Step 3: Select your image
# -----------------------------
print("\nWhich Bollywood Celebrity Are You?")
Tk().withdraw()  # Hide root window
uploaded_image = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
)

if not uploaded_image:
    print("❌ No file selected. Please run again and choose an image.")
    exit()

# Save a copy (optional)
os.makedirs("uploads", exist_ok=True)
save_path = os.path.join("uploads", os.path.basename(uploaded_image))
Image.open(uploaded_image).save(save_path)

# -----------------------------
# Step 4: Find the best celebrity match
# -----------------------------
try:
    result = DeepFace.find(
        img_path=save_path,
        db_path="celebrity_db",  # Path to pre-stored celebrity images
        model_name="VGG-Face",
        enforce_detection=False
    )

    if len(result) > 0 and not result[0].empty:
        best_match = result[0].iloc[0].identity
        predicted_actor = os.path.basename(best_match).replace("_", " ").split(".")[0]
        print(f"\n✅ You look like: {predicted_actor}")

        # Show side-by-side images
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(Image.open(save_path))
        axes[0].set_title("Your Uploaded Image")
        axes[0].axis("off")

        axes[1].imshow(Image.open(best_match))
        axes[1].set_title(f"Seems like {predicted_actor}")
        axes[1].axis("off")

        plt.show()
    else:
        print("❌ No match found. Try another photo.")

except Exception as e:
    print(f"⚠️ Error during celebrity match: {e}")
