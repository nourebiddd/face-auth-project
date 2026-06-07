"""
Experiment 1 — Real Data Authentication
Register on first 10 photos per person, test on held-out last 5.
Same train/test split as Experiment 2 for a fair comparison.
"""
import os
import json
import numpy as np
from deepface import DeepFace

REAL_USERS_DIR = 'data/real_users'
SYNTH_RAW_DIR = 'data/synthetic_raw'
DB_FILE = 'data/registered_users_exp1.json'
MODEL = 'Facenet'
THRESHOLD = 10.0
FRIENDS = ['amr', 'adel', 'otto', 'mazen']

def register_real_users():
    db = {}
    for person in FRIENDS:
        person_dir = os.path.join(REAL_USERS_DIR, person)
        all_imgs = sorted([f for f in os.listdir(person_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))])
        # First 10 only — last 5 held out for test
        train_imgs = all_imgs[:10]
        embeddings = []
        for img_file in train_imgs:
            img_path = os.path.join(person_dir, img_file)
            try:
                emb = DeepFace.represent(img_path, model_name=MODEL, enforce_detection=False)
                embeddings.append(emb[0]['embedding'])
            except:
                pass
        if embeddings:
            db[person] = np.mean(embeddings, axis=0).tolist()
            print(f"  Registered {person} ({len(embeddings)}/10 training photos)")
    with open(DB_FILE, 'w') as f:
        json.dump(db, f)
    print(f"[INFO] Registration saved to {DB_FILE}")
    return db

def authenticate(img_path, db):
    try:
        emb = DeepFace.represent(img_path, model_name=MODEL, enforce_detection=False)
        query = np.array(emb[0]['embedding'])
        best_match, best_score = None, float('inf')
        for person, stored_emb in db.items():
            dist = np.linalg.norm(query - np.array(stored_emb))
            if dist < best_score:
                best_score = dist
                best_match = person
        return best_score < THRESHOLD, best_match, best_score
    except:
        return None, None, None

def run_test(db):
    print('\n' + '='*65)
    print('EXPERIMENT 1 — REAL REGISTRATION, HELD-OUT REAL TEST')
    print('='*65)

    print('\n' + '='*65)
    print('FRIENDS TEST (last 5 photos per person — never used in registration)')
    print('='*65)
    tp, fn = 0, 0
    for person in FRIENDS:
        real_dir = os.path.join(REAL_USERS_DIR, person)
        all_imgs = sorted([f for f in os.listdir(real_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))])
        test_imgs = all_imgs[-5:]
        for img in test_imgs:
            verified, match, score = authenticate(os.path.join(real_dir, img), db)
            if verified is None:
                continue
            status = '✅ VERIFIED' if verified else '❌ REJECTED'
            print(f'{status} | True: {person:<8} | Predicted: {match:<8} | Score: {score:.2f}')
            if verified:
                tp += 1
            else:
                fn += 1

    print('\n' + '='*65)
    print('INTRUDER TEST (20 AI strangers)')
    print('='*65)
    tn, fp = 0, 0
    for i in range(1, 21):
        img_path = os.path.join(SYNTH_RAW_DIR, f'synthetic_person_{i:03d}', 'img_0000.jpg')
        if not os.path.exists(img_path):
            continue
        verified, match, score = authenticate(img_path, db)
        if verified is None:
            continue
        status = '✅ VERIFIED' if verified else '❌ REJECTED'
        print(f'{status} | True: intruder  | Predicted: {match:<8} | Score: {score:.2f}')
        if not verified:
            tn += 1
        else:
            fp += 1

    total_friends = tp + fn
    total_intruders = tn + fp
    far = fp / total_intruders * 100 if total_intruders > 0 else 0
    frr = fn / total_friends * 100 if total_friends > 0 else 0
    acc = (tp + tn) / (total_friends + total_intruders) * 100

    print('\n' + '='*65)
    print('EXPERIMENT 1 RESULTS (fair split)')
    print('='*65)
    print(f'Friends correctly verified  : {tp}/{total_friends}')
    print(f'Friends wrongly rejected    : {fn}/{total_friends}')
    print(f'Intruders correctly rejected: {tn}/{total_intruders}')
    print(f'Intruders wrongly verified  : {fp}/{total_intruders}')
    print(f'FAR (False Accept Rate)     : {far:.1f}%')
    print(f'FRR (False Reject Rate)     : {frr:.1f}%')
    print(f'Accuracy                    : {acc:.1f}%')
    print('='*65)

    result = {
        'experiment': 1,
        'data_type': 'real',
        'train_photos_per_person': 10,
        'test_photos_per_person': 5,
        'n_intruders_tested': total_intruders,
        'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
        'FAR': round(far, 2),
        'FRR': round(frr, 2),
        'accuracy': round(acc, 2)
    }
    with open('results/exp1_fair_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Results saved to results/exp1_fair_results.json')
    return result

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    if os.path.exists(DB_FILE):
        print(f"[INFO] Loading existing registration...")
        with open(DB_FILE) as f:
            db = json.load(f)
    else:
        print("[INFO] Registering from first 10 real photos per person...")
        db = register_real_users()
    run_test(db)
