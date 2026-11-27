"""Convert HEIC test photos in data/test to JPEG (see convert.py)."""
from convert import convert_folder

if __name__ == "__main__":
    convert_folder("data/test")
