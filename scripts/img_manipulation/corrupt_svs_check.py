"""Module for corrupt svs check."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.wsi import WSI
from openslide import OpenSlideUnsupportedFormatError


def process_svs_images(directory_path, img_embedding_backend="plip"):
    """
    Process all .svs images in the given directory.

    This script tries to load each image and handles the following exceptions:
    - OpenSlideUnsupportedFormatError
    - FileNotFoundError
    - Generic exception
    """

    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"[ERROR] The directory does not exist: {directory_path}")
        return

    # Get all .svs files in the directory
    image_paths = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.lower().endswith(".svs")
    ]

    if not image_paths:
        print(f"[INFO] No .svs images found in the directory: {directory_path}")
        return

    i = 0
    # Iterate over all images in the directory
    for image_path in image_paths:
        # print(f"[INFO] Processing image: {image_path}")
        i += 1
        if i % 10 == 0:
            print(f"[INFO] Processed {i} images...")

        try:
            # Try to load the image
            wsi = WSI(image_path, img_embedding_backend=img_embedding_backend)
            # print(f"[SUCCESS] Successfully loaded image: {image_path}")

        except OpenSlideUnsupportedFormatError as e:
            print(f"[SKIP] OpenSlide cannot read: {image_path}. Error: {e}")
            continue
        except FileNotFoundError:
            print(f"[SKIP] Missing file: {image_path}")
            continue
        except Exception as e:
            print(f"[SKIP] Unknown error opening {image_path}: {e}")
            continue

    print("[INFO] Finished processing images.")


# Main Execution
if __name__ == "__main__":
    # Specify the directory where your .svs images are located
    directory_path = "/Volumes/Xbox_HD/data/med_img"  # Change this to the directory containing your .svs files

    # Process the images
    process_svs_images(directory_path)
