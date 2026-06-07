import cv2
import os

# Load trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("encodings/lbph_model.xml")

test_path = "data/test"

os.makedirs("results", exist_ok=True)
os.makedirs("intruder_logs", exist_ok=True)

print("Testing images...\n")

for file in os.listdir(test_path):
    path = os.path.join(test_path, file)
    img = cv2.imread(path)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (200, 200))

    label, confidence = model.predict(gray)

    # Lower confidence = better match
    if confidence < 80:   # tune between 50–100
        result = "AUTHORIZED"
        color = (0, 255, 0)
    else:
        result = "INTRUDER"
        color = (0, 0, 255)
        cv2.imwrite(f"intruder_logs/{file}", img)

    cv2.putText(img, f"{result} ({int(confidence)})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imwrite(f"results/{file}", img)

    print(f"{file} → {result} (confidence: {int(confidence)})")

print("\nDone.")
