import os, shutil, argparse, random
from PIL import Image, ImageEnhance, ImageFilter

def scan_lfw(src):
    dataset = {}
    for person in sorted(os.listdir(src)):
        person_dir = os.path.join(src, person)
        if not os.path.isdir(person_dir):
            continue
        images = [os.path.join(person_dir, f) for f in os.listdir(person_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if images:
            dataset[person] = images
    return dataset

def augment_image(img):
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

def main(args):
    dataset = scan_lfw(args.src)
    filtered = {p: imgs for p, imgs in dataset.items() if len(imgs) >= args.min_original}
    counts = sorted([len(v) for v in dataset.values()], reverse=True)
    print(f"Total people: {len(dataset)} | Range: {min(counts)}-{max(counts)}")
    print(f"People >= {args.min_original} images: {len(filtered)}")
    os.makedirs(args.dst, exist_ok=True)
    for i, (person, images) in enumerate(filtered.items(), 1):
        dst_dir = os.path.join(args.dst, person)
        os.makedirs(dst_dir, exist_ok=True)
        count = 0
        for img_path in images:
            shutil.copy2(img_path, os.path.join(dst_dir, f"{person}_{count:04d}.jpg"))
            count += 1
        aug = 0
        while count < args.target:
            try:
                img = Image.open(random.choice(images)).convert("RGB")
                augment_image(img).save(os.path.join(dst_dir, f"{person}_aug_{aug:04d}.jpg"), quality=92)
                count += 1; aug += 1
            except: pass
        print(f"[{i}/{len(filtered)}] {person:<40} {len(images)} orig → {count} total")
    print(f"\nDone! Saved to {args.dst}")
    print(f"Next: python train_real.py --data_dir {args.dst} --out_dir results/real")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/lfw-deepfunneled")
    parser.add_argument("--dst", default="data/real_training")
    parser.add_argument("--min_original", type=int, default=10)
    parser.add_argument("--target", type=int, default=30)
    main(parser.parse_args())
