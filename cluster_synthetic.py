import os, pickle, shutil
import numpy as np
import face_recognition
from sklearn.cluster import KMeans

src = "data/synthetic_raw"
dst = "data/synthetic_training"
n_clusters = 62

print("[1/3] Encoding all synthetic faces...")
encodings, paths = [], []
for person in sorted(os.listdir(src)):
    person_dir = os.path.join(src, person)
    if not os.path.isdir(person_dir): continue
    for img in os.listdir(person_dir):
        if not img.lower().endswith((".jpg",".png")): continue
        try:
            image = face_recognition.load_image_file(os.path.join(person_dir, img))
            encs = face_recognition.face_encodings(image)
            if encs:
                encodings.append(encs[0])
                paths.append(os.path.join(person_dir, img))
        except: pass

print(f"[INFO] Encoded {len(encodings)} faces")

print("[2/3] Clustering into 62 identities...")
X = np.array(encodings)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

print("[3/3] Saving clustered identities...")
os.makedirs(dst, exist_ok=True)
counts = [0] * n_clusters
for path, label in zip(paths, labels):
    person_dir = os.path.join(dst, f"synthetic_person_{label+1:03d}")
    os.makedirs(person_dir, exist_ok=True)
    ext = os.path.splitext(path)[1]
    dst_path = os.path.join(person_dir, f"img_{counts[label]:04d}{ext}")
    shutil.copy2(path, dst_path)
    counts[label] += 1

for i, c in enumerate(counts):
    print(f"  synthetic_person_{i+1:03d}: {c} images")

print(f"\nDone! Saved to {dst}")
print("Next: python train_synthetic.py --data_dir data/synthetic_training --out_dir results/synthetic")
