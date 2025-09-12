# !pip install mtcnn==0.1.0
# !pip install tensorflow==2.3.1
# !pip install keras==2.4.3
# !pip install keras-vggface==0.6
# !pip install keras_applications==1.0.8

import os
import pickle

actors = os.listdir('D:/all/which-bollywood-celebrity-are-you-main/data/Bollywood_celeb_face_localized/bollywood_celeb_faces_0')

filenames = []
for actor in actors:
    for file in os.listdir(os.path.join('D:/all/which-bollywood-celebrity-are-you-main/data/Bollywood_celeb_face_localized/bollywood_celeb_faces_0',actor)):
        filenames.append(os.path.join('D:/all/which-bollywood-celebrity-are-you-main/data/Bollywood_celeb_face_localized/bollywood_celeb_faces_0',actor,file))

pickle.dump(filenames,open('filenames.pkl','wb'))
import numpy as np
import pickle
from tqdm import tqdm
from deepface import DeepFace

# Load filenames
with open('filenames.pkl', 'rb') as f:
    filenames = pickle.load(f)

def feature_extractor(img_path):
    """
    Extract deep features from an image using DeepFace VGG-Face.
    
    Args:
        img_path (str): Path to the image.
    
    Returns:
        np.ndarray: Feature vector.
    """
    embedding = DeepFace.represent(img_path=img_path, model_name='VGG-Face', enforce_detection=False)
    return np.array(embedding[0]["embedding"])

# Extract features for all images
features = [feature_extractor(file) for file in tqdm(filenames)]

# Save embeddings
with open('embedding.pkl', 'wb') as f:
    pickle.dump(features, f)

print("Feature extraction completed and saved to 'embedding.pkl'.")
