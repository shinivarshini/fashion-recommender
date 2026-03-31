import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from tqdm import tqdm

# Load the AI model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Get all image paths from your data/images folder
image_folder = 'data/images'
filenames = []

if os.path.exists(image_folder):
    for file in os.listdir(image_folder):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            filenames.append(os.path.join(image_folder, file))
else:
    print(f"Error: The folder {image_folder} does not exist!")

# Run extraction
feature_list = []
if filenames:
    print(f"Extracting features for {len(filenames)} images...")
    for file in tqdm(filenames):
        feature_list.append(extract_features(file, model))

    # Save the "Brain" files
    pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
    pickle.dump(filenames, open('filenames.pkl', 'wb'))
    print("Success! Brain files created.")
else:
    print("No images found in data/images!")