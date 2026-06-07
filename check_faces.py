import face_recognition, os

for person in ['amr','otto','mazen','adel']:
    folder = f'data/real_users/{person}'
    detected = 0
    for img in os.listdir(folder):
        if img.lower().endswith('.jpg'):
            try:
                image = face_recognition.load_image_file(f'{folder}/{img}')
                encs = face_recognition.face_encodings(image)
                if encs: detected += 1
            except: pass
    print(f'{person}: {detected}/15 faces detected')
