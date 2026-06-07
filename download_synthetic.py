import os, time, requests, random
from pathlib import Path

def download_synthetic_faces(out_dir="data/synthetic_raw", n_identities=62, images_per_identity=30):
    os.makedirs(out_dir, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    total = 0
    
    for identity in range(1, n_identities + 1):
        identity_dir = os.path.join(out_dir, f"synthetic_person_{identity:03d}")
        os.makedirs(identity_dir, exist_ok=True)
        count = 0
        attempts = 0
        
        print(f"[{identity}/{n_identities}] Downloading person {identity:03d}...")
        
        while count < images_per_identity and attempts < images_per_identity * 3:
            try:
                r = requests.get("https://thispersondoesnotexist.com/", 
                               headers=headers, timeout=10)
                if r.status_code == 200:
                    img_path = os.path.join(identity_dir, f"img_{count:04d}.jpg")
                    with open(img_path, "wb") as f:
                        f.write(r.content)
                    count += 1
                    total += 1
                time.sleep(random.uniform(1.0, 2.0))
            except Exception as e:
                print(f"  [WARN] {e}")
                time.sleep(2)
            attempts += 1
        
        print(f"  Done: {count} images saved")
    
    print(f"\nTotal downloaded: {total} images across {n_identities} identities")
    print(f"Saved to: {out_dir}")

if __name__ == "__main__":
    download_synthetic_faces()
