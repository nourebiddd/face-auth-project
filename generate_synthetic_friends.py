"""
Experiment 2 — Synthetic Face Generation
Method: identity-preserving augmentation pipeline that simulates controlled
synthetic imaging conditions (lighting, colour temperature, pose, skin tone)
as described in NIST FRTE evaluation guidelines.
"""
import cv2
import numpy as np
import os
import random
import shutil
from insightface.app import FaceAnalysis

FRIENDS = ['amr', 'adel', 'otto', 'mazen']
REAL_DIR = 'data/real_users'
OUTPUT_DIR = 'data/synthetic_friends'
IMAGES_PER_PERSON = 15

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

def get_aligned_face(img_path, out_size=256):
    """Detect, align and crop face to canonical frontal pose."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    face = faces[0]

    # 5 keypoints: left eye, right eye, nose, left mouth, right mouth
    kps = face.kps.astype(np.float32)
    # canonical target positions for a 256x256 output
    dst = np.array([
        [90.0,  90.0],
        [166.0, 90.0],
        [128.0, 140.0],
        [96.0,  190.0],
        [160.0, 190.0],
    ], dtype=np.float32)
    tform, _ = cv2.estimateAffinePartial2D(kps, dst, method=cv2.LMEDS)
    if tform is None:
        return None
    aligned = cv2.warpAffine(img, tform, (out_size, out_size),
                              borderMode=cv2.BORDER_REFLECT)
    return aligned

def simulate_lighting(img):
    """Simulate different lighting conditions."""
    gamma = random.uniform(0.55, 1.6)
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)

def simulate_color_temperature(img):
    """Shift colour temperature — warm (golden hour) or cool (overcast)."""
    b, g, r = cv2.split(img.astype(np.float32))
    mode = random.choice(['warm', 'cool', 'neutral'])
    if mode == 'warm':
        r = np.clip(r * random.uniform(1.05, 1.20), 0, 255)
        b = np.clip(b * random.uniform(0.80, 0.95), 0, 255)
    elif mode == 'cool':
        b = np.clip(b * random.uniform(1.05, 1.20), 0, 255)
        r = np.clip(r * random.uniform(0.80, 0.95), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)

def simulate_skin_tone(img):
    """Normalise skin tone via adaptive histogram equalisation."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=random.uniform(1.5, 3.5),
                             tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def simulate_camera(img):
    """Simulate different camera quality — blur + noise."""
    if random.random() < 0.5:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    noise = np.random.normal(0, random.uniform(2, 8), img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def simulate_pose(img):
    """Slight random in-plane rotation — ±12 degrees."""
    angle = random.uniform(-12, 12)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def generate_synthetic(aligned_face):
    """Apply the full synthetic imaging pipeline."""
    img = aligned_face.copy()
    # Apply each transformation with some randomness so every output differs
    img = simulate_lighting(img)
    img = simulate_color_temperature(img)
    img = simulate_skin_tone(img)
    img = simulate_camera(img)
    img = simulate_pose(img)
    return img

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    for person in FRIENDS:
        real_dir = os.path.join(REAL_DIR, person)
        out_dir = os.path.join(OUTPUT_DIR, person)
        os.makedirs(out_dir, exist_ok=True)

        all_imgs = sorted([f for f in os.listdir(real_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))])
        # Use first 10 only — last 5 are held out as the test set
        real_imgs = all_imgs[:10]
        print(f"\n[{person.upper()}] Using {len(real_imgs)}/15 photos for generation (last 5 held out for testing)...")

        generated = 0
        idx = 0
        while generated < IMAGES_PER_PERSON:
            src = real_imgs[idx % len(real_imgs)]
            idx += 1
            aligned = get_aligned_face(os.path.join(real_dir, src))
            if aligned is None:
                continue
            synthetic = generate_synthetic(aligned)
            out_path = os.path.join(out_dir, f"syn_{generated:04d}.jpg")
            cv2.imwrite(out_path, synthetic)
            generated += 1

        print(f"  Generated {generated}/{IMAGES_PER_PERSON} synthetic images for {person}")

    print(f"\n[DONE] Synthetic friends saved to {OUTPUT_DIR}/")
    print("Controlled variables: lighting (gamma), colour temperature, skin tone (CLAHE),")
    print("camera quality (blur+noise), pose (rotation ±12°)")
    print("\nNext: python exp2_test.py")

if __name__ == "__main__":
    main()
