# 🎯 FaceAttend Pro AI

AI-Powered Face Recognition Attendance System built using OpenCV, PCA, Logistic Regression, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

---

## 📌 Project Overview

FaceAttend Pro AI is a real-time face recognition and attendance management system that automatically identifies individuals using facial features and records attendance. The system also includes gender detection and a modern Streamlit-based dashboard for user interaction.

This project demonstrates practical applications of:

- Machine Learning
- Computer Vision
- Face Recognition
- PCA Dimensionality Reduction
- Logistic Regression Classification
- Real-Time Webcam Processing

---

## 🚀 Features

### 👤 Face Recognition
- Detects faces using OpenCV Haar Cascade
- Identifies registered users in real-time
- Supports multiple individuals

### 🚻 Gender Detection
- Predicts gender from facial images
- Uses PCA + Logistic Regression

### 📋 Attendance Management
- Automatically records attendance
- Prevents duplicate entries
- Exports attendance data as CSV

### 📷 Image Upload Prediction
- Upload image for face recognition
- Detect multiple faces in a single image

### 🧠 Model Training
- Train Face Recognition Model
- Train Gender Detection Model
- Save trained models using Joblib

### 🎨 Interactive Dashboard
- Built with Streamlit
- Modern dark UI
- Real-time monitoring

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Core Programming |
| OpenCV | Face Detection & Image Processing |
| NumPy | Numerical Computing |
| Pandas | Data Handling |
| Scikit-Learn | Machine Learning |
| PCA | Dimensionality Reduction |
| Logistic Regression | Classification |
| Joblib | Model Persistence |
| Streamlit | Web Application |

---

## 📂 Project Structure

```text
FaceAttend-Pro-AI/
│
├── app.py
├── train_face_model.py
├── train_gender_model.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── Person_1/
│   ├── Person_2/
│   └── ...
│
├── models/
│   ├── name_model_clf.pkl
│   ├── name_pca_clf.pkl
│   ├── gender_model.pkl
│   └── gender_clf_pca.pkl
│
├── attendance.csv
│
└── screenshots/
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/faceattend-pro-ai.git

cd faceattend-pro-ai
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 🧠 Machine Learning Pipeline

### Face Recognition

1. Image Collection
2. Face Preprocessing
3. Grayscale Conversion
4. Flattening
5. Normalization
6. PCA Transformation
7. Logistic Regression Training
8. Real-Time Prediction

### Gender Detection

1. Dataset Preparation
2. Image Preprocessing
3. PCA Feature Extraction
4. Logistic Regression Classification
5. Real-Time Inference

---

## 📊 Model Details

### Face Recognition Model

- Algorithm: Logistic Regression
- Dimensionality Reduction: PCA (99.9% Variance)
- Input Size: 150 × 150 Grayscale Images

### Gender Detection Model

- Algorithm: Logistic Regression
- Dimensionality Reduction: PCA (99.9% Variance)
- Classes:
  - Male
  - Female

---

## 📸 Screenshots

### Dashboard
_Add Screenshot Here_

### Face Registration
_Add Screenshot Here_

### Model Training
_Add Screenshot Here_

### Live Face Recognition
_Add Screenshot Here_

### Attendance Dashboard
_Add Screenshot Here_

---

## 🔮 Future Improvements

- Deep Learning Face Recognition
- FaceNet Integration
- Mobile Application
- Cloud Database Support
- Multi-Camera Support
- Employee Management System
- Real-Time Analytics Dashboard

---

## 👨‍💻 Developer

**Chandan Kumar Sah**

GitHub:
https://github.com/ChankumarSah

LinkedIn:
https://www.linkedin.com/in/chandan-kumar-sah-752803387

---

## ⭐ If you found this project useful, consider giving it a star.
