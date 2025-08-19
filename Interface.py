import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle
import cv2

# Load the trained EfficientNetB3 feature extractor
effnet_model = EfficientNetB3(weights="imagenet", include_top=False)
feature_extractor = tf.keras.Model(inputs=effnet_model.input, outputs=effnet_model.get_layer("top_conv").output)

# Load trained models from Google Drive
with open('/content/drive/MyDrive/svm_model.pkl', 'rb') as f:
    best_svm = pickle.load(f)

with open('/content/drive/MyDrive/xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('/content/drive/MyDrive/stacking_model.pkl', 'rb') as f:
    stacking_model = pickle.load(f)

# Load MinMaxScaler and PCA
with open('/content/drive/MyDrive/minmax_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('/content/drive/MyDrive/pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)

# Define Image Preprocessing Functions
IMG_SIZE = (300, 300)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def preprocess_image(image):
    image = cv2.resize(image, IMG_SIZE)
    image = apply_clahe(image)
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Define Prediction Function
def predict_cyberbullying(image):
    # Preprocess Image
    processed_img = preprocess_image(image)

    # Extract Features using EfficientNetB3
    feature_vector = feature_extractor.predict(processed_img)[0].flatten().reshape(1, -1)

    # Apply MinMaxScaler and PCA
    feature_scaled = scaler.transform(feature_vector)
    feature_pca = pca.transform(feature_scaled)

    # Get Prediction from Stacking Model
    final_pred = stacking_model.predict(feature_pca)[0]
    final_label = "Cyberbullying" if final_pred == 1 else "Non-Cyberbullying"

    return final_label

# Create Gradio Interface
interface = gr.Interface(
    fn=predict_cyberbullying,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction"),
    title="Cyberbullying Image Classifier using EfficientNetB3-Metaclassifier",
    description="Upload an image, and the model will predict whether it's cyberbullying or non-cyberbullying.",
    allow_flagging="never"
)

# Launch the interface
interface.launch(debug=True, share=True)
