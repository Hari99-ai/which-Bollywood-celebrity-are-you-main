import os
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image

# Set credentials (or put kaggle.json in ~/.kaggle/)
os.environ['KAGGLE_USERNAME'] = 'hari9931'
os.environ['KAGGLE_KEY'] = 'ebc5af08fae31d15747b946396bb6be3'

# Initialize API
api = KaggleApi()
api.authenticate()

# Example: Download a dataset
dataset = 'hari9931/bollywood-celeb-images'  # replace with actual dataset path
save_path = 'celebrity_db'
os.makedirs(save_path, exist_ok=True)

# Download & unzip
api.dataset_download_files(dataset, path=save_path, unzip=True)

# List downloaded images
for f in os.listdir(save_path):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(save_path, f)
        img = Image.open(img_path)
        img.show()  # open image
        print("Loaded:", f)
