# Bollywood Celebrity Matcher

🎬 **Find Your Bollywood Look-Alike!**  
A fun and interactive web app built with **Streamlit**, **DeepFace**, and **MTCNN** that allows users to upload a photo or take a selfie to find their closest Bollywood celebrity match.

---

## Features

- Upload an image (`.jpg`, `.jpeg`, `.png`, `.webp`) or take a selfie using your webcam.
- Detects face and extracts facial features using **DeepFace (VGG-Face)**.
- Recommends top 3 Bollywood celebrity look-alikes based on **cosine similarity**.
- Displays colorful similarity score bars for each match.
- Shows a congratulatory message if the top match has **similarity ≥ 80%**.
- Handles webp image conversion automatically for compatibility.
- Lightweight and fast, suitable for both local and online deployment.

---

## Tech Stack

- **Python**  
- **Streamlit** – Web app framework for interactive UIs  
- **DeepFace** – Facial recognition and feature extraction  
- **MTCNN** – Face detection  
- **OpenCV** – Image processing  
- **Pillow** – Image handling and webp conversion  
- **scikit-learn** – Cosine similarity computation  
- **Pickle** – Saving and loading embeddings  

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Hari99-ai/Bollywood-Celebrity-Matcher.git
   cd Bollywood-Celebrity-Matcher

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   > If you do not have a `requirements.txt`, install manually:

   ```bash
   pip install streamlit deepface mtcnn opencv-python-headless Pillow scikit-learn
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

---

## Usage

1. Open the Streamlit app in your browser.
2. Choose **Upload Image** or **Use Webcam**.
3. Upload your photo or take a selfie.
4. Wait for the app to analyze your face.
5. View the top 3 Bollywood celebrity matches with colorful similarity scores.
6. If similarity ≥ 80%, see a special congratulatory message.

---


---
## Project Structure

```
Bollywood-Celebrity-Matcher/
│
├─ app.py                  # Main Streamlit app
├─ embedding.pkl           # Precomputed embeddings for celebrity dataset
├─ successful_filenames.pkl# List of image filenames corresponding to embeddings
├─ uploads/                # Folder to store uploaded images
├─ README.md
└─ requirements.txt        # Python dependencies
```
# Dataset
A streamlit web app which can tell with which bollywood celebrity you face resembles
Dataset=https://www.kaggle.com/datasets/sushilyadav1998/bollywood-celeb-localized-face-dataset 
---

## Author

**Hari Om**
❤️ Developed with Python and Streamlit

---

## License

This project is **open-source** and free to use.

```

---

I can also **create a ready-to-use `requirements.txt`** for this project so it runs without TensorFlow/Protobuf issues.  

Do you want me to do that next?
```


