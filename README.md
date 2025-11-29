# Cyberbullying Image Classifier using EfficientNetB3 + Meta-Ensemble (SVM + XGBoost + Stacking)

This project predicts **Cyberbullying vs Non-Cyberbullying** in images** using **deep learning and machine learning** approaches.  
The system leverages **EfficientNetB3** as a feature extractor, followed by an **ensemble classifier (SVM + XGBoost + Ridge Classifier)** to improve detection accuracy.  

---

## 📌 Features
- 🖼️ **Image preprocessing** with CLAHE to enhance contrast.
- 🔎 **Feature extraction** using EfficientNetB3 pretrained on ImageNet.
- 📊 **Dimensionality reduction** using PCA to optimize feature space.
- ⚖️ **Class balancing** using weighted loss functions.
- 🤖 **Ensemble learning** with SVM and XGBoost combined via stacking classifier.
- 📈 **Performance evaluation** with classification reports & accuracy scores.
- 🎛️ **Interactive Gradio interface** for real-time image testing.

---

## 🏗️ Model Pipeline
1. **Input Image** → Preprocessing (CLAHE + resizing to 300x300).  
2. **EfficientNetB3 Feature Extraction** → Converts image to deep feature vector.  
3. **MinMaxScaler + PCA** → Normalization & dimensionality reduction.  
4. **Ensemble Classifier** → SVM + XGBoost with Ridge Classifier meta-learner.  
5. **Prediction** → Outputs `Cyberbullying` or `Not Cyberbullying`.

---

## 📊 Results
- Achieved **99.17% accuracy** across both datasets.  
- Handles complex cyberbullying patterns in image content.


