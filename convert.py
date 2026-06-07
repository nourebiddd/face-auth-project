from PIL import Image
import pillow_heif
import os

pillow_heif.register_heif_opener()

input_folder = "data/real_user"

for file in os.listdir(input_folder):
    if file.endswith(".HEIC"):
        path = os.path.join(input_folder, file)
        
        image = Image.open(path)
        new_name = file.replace(".HEIC", ".jpg")
        new_path = os.path.join(input_folder, new_name)
        
        image.save(new_path, "JPEG")
        print(f"Converted: {file} → {new_name}")
