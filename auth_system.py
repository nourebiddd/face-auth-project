import os
import json
import numpy as np
from deepface import DeepFace

DB_FILE = "data/registered_users.json"
MODEL = "Facenet"
THRESHOLD = 10.0

def register_users(users_dir="data/real_users"):
    """Register all friends into the system."""
    db = {}
    people = [p for p in os.listdir(users_dir) if os.path.isdir(os.path.join(users_dir, p))]
    print(f"[INFO] Registering {len(people)} users...")
    for person in people:
        person_dir = os.path.join(users_dir, person)
        embeddings = []
        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith(".jpg"): continue
            img_path = os.path.join(person_dir, img_file)
            try:
                emb = DeepFace.represent(img_path, model_name=MODEL, enforce_detection=False)
                embeddings.append(emb[0]["embedding"])
            except: pass
        if embeddings:
            db[person] = np.mean(embeddings, axis=0).tolist()
            print(f"  ✅ {person} registered ({len(embeddings)} images)")
    with open(DB_FILE, "w") as f:
        json.dump(db, f)
    print(f"\n[INFO] {len(db)} users saved to {DB_FILE}")

def authenticate(img_path):
    """Check if a face is a registered user or intruder."""
    with open(DB_FILE, "r") as f:
        db = json.load(f)
    try:
        emb = DeepFace.represent(img_path, model_name=MODEL, enforce_detection=False)
        query = np.array(emb[0]["embedding"])
    except:
        print("❌ No face detected in image")
        return

    best_match = None
    best_score = float("inf")
    for person, stored_emb in db.items():
        stored = np.array(stored_emb)
        dist = np.linalg.norm(query - stored)
        if dist < best_score:
            best_score = dist
            best_match = person

    print(f"\n{'='*40}")
    if best_score < THRESHOLD:
        print(f"  ✅ VERIFIED — Welcome {best_match}!")
        print(f"  Confidence score: {best_score:.2f}")
    else:
        print(f"  ❌ INTRUDER ALERT — Access Denied!")
        print(f"  Closest match: {best_match} (score: {best_score:.2f})")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Registering users...")
        register_users()
    else:
        print(f"Authenticating: {sys.argv[1]}")
        authenticate(sys.argv[1])
