import os
import cv2
import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# ==========================================
# DATASET PATHS
# ==========================================

MALE_PATH = "dataset/male"
FEMALE_PATH = "dataset/female"

# ==========================================
# LOAD IMAGES FUNCTION
# ==========================================

def load_images(folder_path, label):

    features = []
    labels = []

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.resize(image, (150, 150))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = image.flatten() / 255.0

        features.append(image)
        labels.append(label)

    return features, labels


# ==========================================
# LOAD MALE IMAGES
# ==========================================

X_male, y_male = load_images(MALE_PATH, "male")

# ==========================================
# LOAD FEMALE IMAGES
# ==========================================

X_female, y_female = load_images(FEMALE_PATH, "female")

# ==========================================
# COMBINE DATA
# ==========================================

X = np.array(X_male + X_female)
y = np.array(y_male + y_female)

print(f"Dataset Shape : {X.shape}")

# ==========================================
# PCA
# ==========================================

gender_clf_pca = PCA(n_components=0.999)

X_gender = gender_clf_pca.fit_transform(X)

print(f"Original Features : {X.shape[1]}")
print(f"PCA Components    : {X_gender.shape[1]}")

# ==========================================
# MODEL TRAINING
# ==========================================

gender_dect_model = LogisticRegression(
    max_iter=5000,
    random_state=42
)

gender_dect_model.fit(X_gender, y)

accuracy = gender_dect_model.score(X_gender, y)

print(f"Training Accuracy : {accuracy:.4f}")

# ==========================================
# SAVE MODELS
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    gender_dect_model,
    "models/gender_model.pkl"
)

joblib.dump(
    gender_clf_pca,
    "models/gender_clf_pca.pkl"
)

print("✅ Gender Model Saved Successfully")
