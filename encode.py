import cv2
import os
import numpy as np

data_path = "data/real_user"

images = []
labels = []

label_id = 0  # only one person (you)

print("Preparing training data...")

for file in os.listdir(data_path):
    path = os.path.join(data_path, file)
    img = cv2.imread(path)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (200, 200))

    images.append(gray)
    labels.append(label_id)

print(f"Loaded {len(images)} images")

# Train LBPH model
model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
model.train(images, np.array(labels))

# Save model
os.makedirs("encodings", exist_ok=True)
model.save("encodings/lbph_model.xml")

print("Training done and model saved.")
