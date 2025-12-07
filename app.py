import os
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D

from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import streamlit as st 
import pickle

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Fashion Recommendation System",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================================
# Load Image Features
# ================================
image_features = pickle.load(open('Images_features.pkl','rb'))
filenames = pickle.load(open('filenames.pkl','rb'))

# ================================
# Feature Extraction Function
# ================================
def extract_features_from_images(image_path, model):
    try:
        img = image.load_img(image_path, target_size=(224,224))
        img_array = image.img_to_array(img)
        img_expand_dim = np.expand_dims(img_array, axis=0)
        img_preprocess = preprocess_input(img_expand_dim)
        result = model.predict(img_preprocess, verbose=0).flatten() 
        
        # Normalize features
        norm_result = result/norm(result) 
        return norm_result
    except UnidentifiedImageError:
        st.error("❌ Invalid image file. Please upload a valid image.")
        return None

# ================================
# Load Model
# ================================

def load_model():
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    model.trainable = False
    model = tf.keras.Sequential([model, GlobalMaxPooling2D()])
    return model

model = load_model()

# ================================
# Nearest Neighbor Model
# ================================
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(image_features)

# ================================
# HEADER
# ================================
st.markdown("<h1 style='text-align:center; color:#ff4b4b;'>👗 Fashion Recommendation System</h1>", unsafe_allow_html=True)
st.write("### Upload an image and get visually similar fashion recommendations.")

# ================================
# SIDEBAR
# ================================
st.sidebar.header("Upload Your Fashion Image")
uploaded_file = st.sidebar.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])

# ================================
# MAIN LOGIC
# ================================

if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("Image uploaded successfully!")

    st.markdown("### 📸 Uploaded Image")
    st.image(uploaded_file,  width=400)

    # Extract features + get recommendations
    with st.spinner("🔍 Finding similar items..."):
        input_img_features = extract_features_from_images(uploaded_file, model)
        distances, indices = neighbors.kneighbors([input_img_features])
    st.markdown("### 🌟 Recommended Items For You")
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(filenames[indices[0][1]])
    with col2:
        st.image(filenames[indices[0][2]])
    with col3:
        st.image(filenames[indices[0][3]])
    with col4:
        st.image(filenames[indices[0][4]])
    with col5:
        st.image(filenames[indices[0][5]])