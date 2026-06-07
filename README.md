# Face Authentication: Real vs Synthetic Training Data

A research project comparing two face authentication systems — one trained on **real photos** and one trained on **synthetically generated photos** — to evaluate whether synthetic data can replace real biometric data in authentication systems.

## Hypothesis

> A face authentication model registered with real facial data will achieve a lower False Accept Rate (FAR) than a model registered with synthetically generated facial data, within a confidence threshold of 10.0.

- **H₀:** No significant difference in FAR between real and synthetic models
- **H₁:** Real model has lower FAR than synthetic model

---

## Project Structure

```
face-auth-project/
├── auth_system.py          # Main auth system (FaceNet via DeepFace)
├── main.py                 # LBPH-based face recognition
├── encode.py               # Encode faces into embeddings
├── train_real.py           # Train SVM classifier on real data (LFW)
├── train_synthetic.py      # Train SVM classifier on synthetic data
├── exp1_test.py            # Experiment 1: real-data auth test
├── exp2_test.py            # Experiment 2: synthetic-data auth test
├── compare_models.py       # Compare real vs synthetic model metrics
├── det_curve.py            # Plot Detection Error Tradeoff (DET) curve
├── results/                # Metrics, confusion matrices, charts
├── paper/                  # LaTeX research paper source
└── docs/                   # Notes and documentation
```

---

## Models Used

| Component | Model | Type |
|---|---|---|
| Auth system (`auth_system.py`) | FaceNet (Google, 22-layer NN) | Neural Network |
| LFW experiments (`train_real/synthetic.py`) | SVM classifier | Machine Learning |
| Alternative system (`main.py`) | LBPH (OpenCV) | Traditional CV |

**FaceNet** converts faces into 128-dimensional embeddings. A distance < 10.0 from a registered embedding counts as verified.

---

## Experiments

### Experiment 1 — Real Data
- 4 registered users (real photos, multiple images averaged into one embedding)
- Tested on 20 friend images + 10 AI-generated intruder images
- Threshold: 10.0

| Metric | Result |
|---|---|
| Friends verified | 20 / 20 |
| Intruders rejected | 9 / 10 |
| False Accept Rate (FAR) | 10.0% |
| False Reject Rate (FRR) | 0.0% |
| Accuracy | 96.7% |

### Experiment 2 — Synthetic Data
- Same 4 users, same threshold, same intruder set
- Registration uses AI-generated versions of each person's face
- Results compared against Experiment 1

---

## Variables

- **Independent:** Type of training data (real vs synthetic)
- **Dependent:** FAR, FRR, Accuracy
- **Controlled:** Same 4 subjects, same FaceNet model, same threshold (10.0), same photo count, same intruder set

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install deepface opencv-python scikit-learn numpy
```

### Register users and run authentication

```bash
# Register real users
python auth_system.py

# Authenticate a single image
python auth_system.py path/to/face.jpg
```

### Run experiments

```bash
python exp1_test.py   # Experiment 1 (real data)
python exp2_test.py   # Experiment 2 (synthetic data)
python compare_models.py  # Compare results
python det_curve.py       # Plot DET curve
```

---

## Evaluation Framework

Results are evaluated following the [NIST FRTE (Face Recognition Technology Evaluation)](https://pages.nist.gov/frvt/html/frvt11.html) framework, using FAR and FRR as primary metrics.

---

## Paper

The full research paper (LaTeX source) is in the `paper/` directory. It documents the methodology, results, and analysis of both experiments.
