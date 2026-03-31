import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# 1. Load the "Saved Memory"
# These files must be in your GitHub repository
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

# 2. Load the Model (The Brain) - Optimized for Cloud
@st.cache_resource
def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    return tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

model = load_model()

st.title('Fashion Recommender System')

# 3. Helper Function to extract features from uploaded image
def extract_feature(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# 4. The Recommender Logic
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# 5. The Website Interface
uploaded_file = st.file_uploader("Upload an item of clothing")

if uploaded_file is not None:
    # Save the file temporarily
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show the uploaded image
    display_image = Image.open(uploaded_file)
    st.image(display_image, width=200, caption='Your Selection')

    # Process and Recommend
    features = extract_feature(os.path.join("uploads", uploaded_file.name), model)
    indices = recommend(features, feature_list)

    # Display the results in 5 columns
    st.subheader("Recommended for you:")
    col1, col2, col3, col4, col5 = st.columns(5)

    # Map back to your data folder
    # Note: Ensure the filenames in your .pkl match the folder on GitHub
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