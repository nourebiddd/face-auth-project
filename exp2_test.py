import os
import json
import numpy as np
from deepface import DeepFace

SYNTH_USERS_DIR = 'data/synthetic_friends'
DB_FILE = 'data/registered_users_synthetic.json'
REAL_USERS_DIR = 'data/real_users'
SYNTH_RAW_DIR = 'data/synthetic_raw'
MODEL = 'Facenet'
THRESHOLD = 10.0
FRIENDS = ['amr', 'adel', 'otto', 'mazen']

def register_synthetic_users():
    db = {}
    people = [p for p in os.listdir(SYNTH_USERS_DIR)
              if os.path.isdir(os.path.join(SYNTH_USERS_DIR, p))]
    print(f"[INFO] Registering {len(people)} synthetic users...")
    for person in people:
        person_dir = os.path.join(SYNTH_USERS_DIR, person)
        embeddings = []
        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(person_dir, img_file)
            try:
                emb = DeepFace.represent(img_path, model_name=MODEL, enforce_detection=False)
                embeddings.append(emb[0]['embedding'])
            except:
                pass
        if embeddings:
            db[person] = np.mean(embeddings, axis=0).tolist()
            print(f"  Registered {person} ({len(embeddings)} synthetic images)")
    with open(DB_FILE, 'w') as f:
        json.dump(db, f)
    print(f"\n[INFO] Synthetic registration saved to {DB_FILE}")
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
        verified = best_score < THRESHOLD
        return verified, best_match, best_score
    except:
        return None, None, None

def run_test(db):
    print('\n' + '='*65)
    print('EXPERIMENT 2 — SYNTHETIC REGISTRATION, REAL TEST')
    print('='*65)

    print('\n' + '='*65)
    print('FRIENDS TEST — registered from SYNTHETIC, tested on REAL (held-out)')
    print('='*65)
    tp, fn = 0, 0
    for person in FRIENDS:
        real_dir = os.path.join(REAL_USERS_DIR, person)
        all_imgs = sorted([f for f in os.listdir(real_dir)
                           if f.lower().endswith(('.jpg','.jpeg','.png','.JPG'))])
        # Last 5 photos — never used during synthetic generation
        imgs = all_imgs[-5:]
        for img in imgs:
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
    acc = (tp + tn) / (total_friends + total_intruders) * 100 if (total_friends + total_intruders) > 0 else 0

    print('\n' + '='*65)
    print('EXPERIMENT 2 RESULTS')
    print('='*65)
    print(f'Friends correctly verified  : {tp}/{total_friends}')
    print(f'Friends wrongly rejected    : {fn}/{total_friends}')
    print(f'Intruders correctly rejected: {tn}/{total_intruders}')
    print(f'Intruders wrongly verified  : {fp}/{total_intruders}')
    print(f'FAR (False Accept Rate)     : {far:.1f}%')
    print(f'FRR (False Reject Rate)     : {frr:.1f}%')
    print(f'Accuracy                    : {acc:.1f}%')
    print('='*65)
    print('\nEXPERIMENT 1 vs EXPERIMENT 2 COMPARISON')
    print('='*65)
    print(f'{"Metric":<30} {"Exp 1 (Real)":<20} {"Exp 2 (Synthetic)":<20}')
    print('-'*65)
    print(f'{"FAR (False Accept Rate)":<30} {"10.0%":<20} {far:.1f}%')
    print(f'{"FRR (False Reject Rate)":<30} {"0.0%":<20} {frr:.1f}%')
    print(f'{"Accuracy":<30} {"96.7%":<20} {acc:.1f}%')
    print(f'{"Friends Verified":<30} {"20/20":<20} {tp}/{total_friends}')
    print(f'{"Intruders Rejected":<30} {"9/10":<20} {tn}/{total_intruders}')
    print('='*65)

    gap_far = far - 10.0
    print(f'\nFAR Gap (Synthetic - Real): {gap_far:+.1f}%')
    if abs(gap_far) <= 15.0:
        print('HYPOTHESIS: CONFIRMED — gap within 15% threshold')
        print('Synthetic data is a viable substitute for real data.')
    else:
        print('HYPOTHESIS: REJECTED — gap exceeds 15% threshold')
        print('Real data significantly outperforms synthetic data.')

    result = {
        'experiment': 2,
        'data_type': 'synthetic',
        'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
        'FAR': round(far, 2),
        'FRR': round(frr, 2),
        'accuracy': round(acc, 2),
        'far_gap_vs_exp1': round(gap_far, 2)
    }
    with open('results/exp2_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nResults saved to results/exp2_results.json')

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    if os.path.exists(DB_FILE):
        print(f"[INFO] Loading existing synthetic registration from {DB_FILE}")
        with open(DB_FILE) as f:
            db = json.load(f)
    else:
        db = register_synthetic_users()
    run_test(db)
