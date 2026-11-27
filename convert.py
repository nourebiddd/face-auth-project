"""Convert HEIC photos (iPhone) in a folder to JPEG.

Usage:
    python convert.py                 # converts data/real_user
    python convert.py data/test       # converts any other folder
"""
import os
import sys

from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()


def convert_folder(input_folder):
    for file in os.listdir(input_folder):
        stem, ext = os.path.splitext(file)
        if ext.lower() != ".heic":
            continue

        path = os.path.join(input_folder, file)
        new_name = stem + ".jpg"
        new_path = os.path.join(input_folder, new_name)

        image = Image.open(path).convert("RGB")
        image.save(new_path, "JPEG")
        print(f"Converted: {file} → {new_name}")


if __name__ == "__main__":
    convert_folder(sys.argv[1] if len(sys.argv) > 1 else "data/real_user")
