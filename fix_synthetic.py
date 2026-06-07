import os, shutil, random
from PIL import Image, ImageEnhance, ImageFilter

src = "data/synthetic_raw"
dst = "data/synthetic_training_fair"
target = 30

def augment(img):
    choice = random.randint(0, 5)
    if choice == 0: return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif choice == 1: return img.rotate(random.uniform(-15, 15), fillcolor=(0,0,0))
    elif choice == 2: return ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
    elif choice == 3: return ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    elif choice == 4: return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    else:
        w, h = img.size
        mx, my = int(w*0.1), int(h*0.1)
        return img.crop((mx, my, w-mx, h-my)).resize((w, h), Image.LANCZOS)

os.makedirs(dst, exist_ok=True)
all_images = []
for person in sorted(os.listdir(src)):
    person_dir = os.path.join(src, person)
    if not os.path.isdir(person_dir): continue
    for img_file in os.listdir(person_dir):
        if img_file.lower().endswith((".jpg",".png")):
            all_images.append(os.path.join(person_dir, img_file))

print(f"[INFO] Found {len(all_images)} synthetic faces")
print(f"[INFO] Creating 62 identities with {target} images each...")

for i, img_path in enumerate(all_images[:62], 1):
    identity_dir = os.path.join(dst, f"synthetic_person_{i:03d}")
    os.makedirs(identity_dir, exist_ok=True)
    shutil.copy2(img_path, os.path.join(identity_dir, "img_0000.jpg"))
    aug_count = 1
    img = Image.open(img_path).convert("RGB")
    while aug_count < target:
        aug = augment(img)
        aug.save(os.path.join(identity_dir, f"img_{aug_count:04d}.jpg"), quality=92)
        aug_count += 1
    print(f"  [{i}/62] synthetic_person_{i:03d}: {target} images")

print(f"\nDone! Saved to {dst}")
print(f"Next: python train_synthetic.py --data_dir {dst} --out_dir results/synthetic")
