import os

import face_recognition

for person in ['amr', 'otto', 'mazen', 'adel']:
    folder = f'data/real_users/{person}'
    images = [img for img in os.listdir(folder) if img.lower().endswith('.jpg')]
    detected = 0
    for img in images:
        try:
            image = face_recognition.load_image_file(f'{folder}/{img}')
            if face_recognition.face_encodings(image):
                detected += 1
        except Exception:
            pass
    print(f'{person}: {detected}/{len(images)} faces detected')
