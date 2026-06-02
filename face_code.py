import os
import cv2
import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# ==========================================
# DATASET PATH
# ==========================================

DATASET_PATH = "dataset"

# ==========================================
# LOAD DATASET
# ==========================================

X = []
y = []

for person_name in os.listdir(DATASET_PATH):

    person_folder = os.path.join(
        DATASET_PATH,
        person_name
    )

    if not os.path.isdir(person_folder):
        continue

    print(f"Loading: {person_name}")

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(
            person_folder,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.resize(
            image,
            (150, 150)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        image = image.flatten() / 255.0

        X.append(image)
        y.append(person_name)

# ==========================================
# CONVERT TO NUMPY
# ==========================================

X = np.array(X)
y = np.array(y)

print(f"Dataset Shape : {X.shape}")
print(f"Classes       : {np.unique(y)}")
print(f"Total Classes : {len(np.unique(y))}")

# ==========================================
# PCA
# ==========================================

name_pca_clf = PCA(
    n_components=0.999
)

X_name = name_pca_clf.fit_transform(X)

print(f"Original Features : {X.shape[1]}")
print(f"PCA Components    : {X_name.shape[1]}")

# ==========================================
# LOGISTIC REGRESSION
# ==========================================

face_clf_model = LogisticRegression(
    max_iter=5000,
    random_state=42
)

face_clf_model.fit(
    X_name,
    y
)

accuracy = face_clf_model.score(
    X_name,
    y
)

print(f"Training Accuracy : {accuracy:.4f}")

# ==========================================
# SAVE MODELS
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    face_clf_model,
    "models/name_model_clf.pkl"
)

joblib.dump(
    name_pca_clf,
    "models/name_pca_clf.pkl"
)

print("✅ Face Recognition Model Saved Successfully")
