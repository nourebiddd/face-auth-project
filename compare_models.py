import os, pickle, numpy as np, json
import face_recognition
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ── Load real model ──
print("[1/4] Loading real model...")
with open("results/real/model_real.pkl", "rb") as f:
    real_model = pickle.load(f)
real_clf = real_model["classifier"]
real_le  = real_model["label_encoder"]

# ── Load LFW test set (people with 20+ images) ──
print("[2/4] Encoding LFW test faces...")
test_dir = "data/real_training"
encodings, labels, names = [], [], []

for person in sorted(os.listdir(test_dir)):
    person_dir = os.path.join(test_dir, person)
    if not os.path.isdir(person_dir): continue
    images = [f for f in os.listdir(person_dir)
              if f.lower().endswith((".jpg",".jpeg",".png"))]
    # use last 6 images as test set (unseen)
    for img_file in images[-6:]:
        try:
            img  = face_recognition.load_image_file(os.path.join(person_dir, img_file))
            encs = face_recognition.face_encodings(img)
            if encs:
                encodings.append(encs[0])
                labels.append(person)
        except: pass

print(f"[INFO] Test set: {len(encodings)} faces from {len(set(labels))} people")
X_test = np.array(encodings)

# ── Test real model on LFW ──
print("[3/4] Testing real model on LFW...")
known = [l for l in labels if l in real_le.classes_]
known_idx = [i for i, l in enumerate(labels) if l in real_le.classes_]
X_known = X_test[known_idx]
y_known = real_le.transform([labels[i] for i in known_idx])
y_pred_real = real_clf.predict(X_known)
acc_real = accuracy_score(y_known, y_pred_real)
print(f"  Real model accuracy on LFW test: {acc_real*100:.2f}%")

# ── Load synthetic model ──
print("[4/4] Testing synthetic model on LFW...")
with open("results/synthetic/model_synthetic.pkl", "rb") as f:
    syn_model = pickle.load(f)
syn_clf = syn_model["classifier"]
syn_le  = syn_model["label_encoder"]

# synthetic model predicts synthetic labels — measure confidence instead
probs = syn_clf.predict_proba(X_known)
max_conf_real = float(np.mean(np.max(probs, axis=1)))

# reload synthetic summary
with open("results/real/summary_real.json") as f:
    real_summary = json.load(f)
with open("results/synthetic/summary_synthetic.json") as f:
    syn_summary = json.load(f)

# ── Plot comparison ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Real vs Synthetic Training Data — Face Authentication", fontsize=14, fontweight="bold")

# Bar chart — accuracy
models  = ["Real Data\n(LFW)", "Synthetic Data\n(StyleGAN2)"]
accuracies = [real_summary["accuracy"]*100, syn_summary["accuracy"]*100]
colors  = ["#2d6a4f", "#7b2d8b"]
bars = axes[0].bar(models, accuracies, color=colors, width=0.4)
axes[0].set_ylim(0, 110)
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Training Accuracy Comparison")
for bar, acc in zip(bars, accuracies):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{acc:.2f}%", ha="center", fontweight="bold")

# Bar chart — samples
samples = [real_summary["n_samples"], syn_summary["n_samples"]]
bars2 = axes[1].bar(models, samples, color=colors, width=0.4)
axes[1].set_ylabel("Number of Training Samples")
axes[1].set_title("Training Dataset Size")
for bar, s in zip(bars2, samples):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(s), ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("results/comparison.png", dpi=150)
print("\n" + "="*55)
print("  FINAL COMPARISON RESULTS")
print("="*55)
print(f"  Real Data Model Accuracy    : {real_summary['accuracy']*100:.2f}%")
print(f"  Synthetic Data Model Accuracy: {syn_summary['accuracy']*100:.2f}%")
print(f"  Gap                         : {(real_summary['accuracy']-syn_summary['accuracy'])*100:.2f}%")
print("="*55)
print(f"\n  Comparison chart saved → results/comparison.png")
