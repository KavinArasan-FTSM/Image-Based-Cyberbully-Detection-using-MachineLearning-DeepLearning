import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
import cv2
import pickle

# Set dataset path
DATASET_PATH = "/content/drive/MyDrive/Dataset"
YES_PATH = os.path.join(DATASET_PATH, "Yes")
NO_PATH = os.path.join(DATASET_PATH, "No")

# Define image size
IMG_SIZE = (300, 300)

# Load EfficientNetB3 Model (without top layers)
base_model = EfficientNetB3(weights="imagenet", include_top=False)

# Extract features from top_conv layer
feature_extractor = tf.keras.Model(
    inputs=base_model.input,
    outputs=base_model.get_layer("top_conv").output
)

# Data Augmentation with CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# Function to load, augment, and preprocess images
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = apply_clahe(img_array.astype(np.uint8))  # Apply CLAHE
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Extract features and labels from dataset
def extract_features_and_labels():
    features, labels = [], []

    for img_name in os.listdir(YES_PATH):
        img_path = os.path.join(YES_PATH, img_name)
        img_array = load_and_preprocess_image(img_path)
        feature_vector = feature_extractor.predict(img_array)[0]
        features.append(feature_vector)
        labels.append(1)

    for img_name in os.listdir(NO_PATH):
        img_path = os.path.join(NO_PATH, img_name)
        img_array = load_and_preprocess_image(img_path)
        feature_vector = feature_extractor.predict(img_array)[0]
        features.append(feature_vector)
        labels.append(0)

    return np.array(features), np.array(labels)

# Extract Features & Labels
X, y = extract_features_and_labels()

# Normalize Features & Save MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))

with open("/content/drive/MyDrive/minmax_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Apply PCA & Save It
pca = PCA(n_components=0.98)
X_pca = pca.fit_transform(X_scaled)

with open("/content/drive/MyDrive/pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

#Compute Class Weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# SVM Hyperparameter Tuning
param_grid = {
    'C': [1, 10, 100, 500],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

svm_grid = GridSearchCV(SVC(probability=True, class_weight=class_weight_dict), param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
svm_grid.fit(X_train, y_train)

# Save Best SVM Model
best_svm = svm_grid.best_estimator_
print(f"\n✅ Best SVM Parameters: {svm_grid.best_params_}")

with open("/content/drive/MyDrive/svm_model.pkl", "wb") as f:
    pickle.dump(best_svm, f)

# Train & Save XGBoost Model
xgb_model = XGBClassifier(n_estimators=500, max_depth=20, learning_rate=0.02, objective='binary:logistic')
xgb_model.fit(X_train, y_train)

with open("/content/drive/MyDrive/xgboost_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

# Train & Save Stacking Ensemble (Using Ridge Classifier)
stacking_model = StackingClassifier(
    estimators=[('svm', best_svm), ('xgb', xgb_model)],
    final_estimator=RidgeClassifier()
)
stacking_model.fit(X_train, y_train)

with open("/content/drive/MyDrive/stacking_model.pkl", "wb") as f:
    pickle.dump(stacking_model, f)

# Evaluate Models
models = {"SVM": best_svm, "XGBoost": xgb_model, "Stacking Ensemble": stacking_model}

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ {name} Model Accuracy: {accuracy * 100:.2f}%")
    print(f"\n{name} Classification Report:\n", classification_report(y_test, y_pred))

print("\n All models, scaler, and PCA have been successfully trained and saved!")
