import os, argparse, pickle, time, numpy as np, json
import face_recognition
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def encode_faces(data_dir):
    encodings, labels = [], []
    people = sorted(os.listdir(data_dir))
    print(f"[INFO] Encoding {len(people)} people...")
    for i, person in enumerate(people, 1):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir): continue
        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith((".jpg",".jpeg",".png")): continue
            try:
                img = face_recognition.load_image_file(os.path.join(person_dir, img_file))
                encs = face_recognition.face_encodings(img)
                if encs:
                    encodings.append(encs[0])
                    labels.append(person)
            except: pass
        print(f"  [{i}/{len(people)}] {person} done")
    print(f"[INFO] Total encoded: {len(encodings)} faces")
    return encodings, labels

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    cache = os.path.join(args.out_dir, "encodings_real.pkl")
    if os.path.exists(cache) and not args.force_encode:
        print(f"[INFO] Loading cached encodings...")
        with open(cache,"rb") as f: encodings, labels = pickle.load(f)
    else:
        encodings, labels = encode_faces(args.data_dir)
        with open(cache,"wb") as f: pickle.dump((encodings, labels), f)

    X = np.array(encodings)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"[INFO] {X.shape[0]} samples | {len(le.classes_)} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    clf = SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, random_state=42)
    print("[INFO] Training SVM...")
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"[INFO] Training done in {time.time()-t0:.2f}s")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    print(f"\n{'='*50}\n  Accuracy (Real Data): {acc*100:.2f}%\n{'='*50}\n")
    print(report)

    with open(os.path.join(args.out_dir,"model_real.pkl"),"wb") as f:
        pickle.dump({"classifier":clf,"label_encoder":le}, f)
    with open(os.path.join(args.out_dir,"metrics_real.txt"),"w") as f:
        f.write(f"Accuracy: {acc*100:.2f}%\n\n{report}")
    with open(os.path.join(args.out_dir,"summary_real.json"),"w") as f:
        json.dump({"data_type":"real","n_samples":int(X.shape[0]),
                   "n_classes":int(len(le.classes_)),"accuracy":float(round(acc,4))}, f, indent=2)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(cm, annot=False, cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix — Real Data")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,"confusion_matrix_real.png"), dpi=150)
    print(f"\n[INFO] All results saved to {args.out_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/real_training")
    parser.add_argument("--out_dir", default="results/real")
    parser.add_argument("--force_encode", action="store_true")
    main(parser.parse_args())
