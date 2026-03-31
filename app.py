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

# Load pre-calculated data
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
# Check if we are in the Cloud (sample_data) or Local (original images)
if os.path.exists('sample_data') and len(os.listdir('sample_data')) > 0:
    filenames = [os.path.join('sample_data', f) for f in os.listdir('sample_data')]
else:
    filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

st.title('👗 Fashion Recommender System')

# Ensure upload directory exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

uploaded_file = st.file_uploader("Choose a clothing image...")

if uploaded_file is not None:
    # Save file
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Show uploaded image
    st.image(Image.open(uploaded_file), width=300, caption="Your Upload")

    # Extract features
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    # Find matches
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([normalized_result])

    # Show matches
    st.header("Recommendations")
    cols = st.columns(5)
    for i in range(1, 6):
        with cols[i-1]:
            st.image(filenames[indices[0][i]])