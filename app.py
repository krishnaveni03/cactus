import streamlit as st
import os
import zipfile
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Reshape, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained model
model = keras.models.load_model('models.h5')


# Create a Streamlit app
st.title("Aerial Cactus Identification")

# Upload an image for prediction
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = plt.imread(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the uploaded image
    img = keras.preprocessing.image.load_img(uploaded_image, target_size=(32, 32))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image

    # Make a prediction
    predictions = model.predict(np.array([img_array]))
    prediction = predictions[0][0]

    # Display the prediction result
    if prediction == 1:
        st.write("Prediction:No Cactus")
    else:
        st.write("Prediction:  Cactus")

st.sidebar.title("About")
st.sidebar.info("This is a Streamlit app for aerial cactus identification.")

# Add any additional information or customization as needed
