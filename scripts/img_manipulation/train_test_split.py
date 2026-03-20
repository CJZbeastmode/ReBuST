import os
import shutil
import random

# Define the source directory containing all images
source_dir = "/Volumes/Xbox_HD/data/med_img"

# Define the target directories for train, val, and test data
val_dir = "/Volumes/Xbox_HD/data/val"
test_dir = "/Volumes/Xbox_HD/data/test"

# Ensure that the target directories exist
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to split the data by categories
def split_data():
    # Get all files in the source directory
    all_files = [f for f in os.listdir(source_dir) if f.endswith(".svs")]

    # Group files by category
    category_files = {}
    for file in all_files:
        # Get the category from the filename (last part after the last hyphen)
        category = file.split("-")[-1].replace(".svs", "")
        
        if category not in category_files:
            category_files[category] = []
        category_files[category].append(file)

    # Split and move files by category
    for category, files in category_files.items():
        print(f"Processing category: {category}")
        # Shuffle the files for randomness
        random.shuffle(files)

        # Calculate the number of files for each split
        total_files = len(files)
        train_count = int(total_files * 0.8)
        val_count = int(total_files * 0.1)
        test_count = total_files - train_count - val_count

        # Split the files into train, validation, and test sets
        train_files = files[:train_count]
        val_files = files[train_count:train_count + val_count]
        test_files = files[train_count + val_count:]

        # Move files to the root of val_dir and test_dir
        for file in val_files:
            shutil.move(os.path.join(source_dir, file), os.path.join(val_dir, file))

        for file in test_files:
            shutil.move(os.path.join(source_dir, file), os.path.join(test_dir, file))

        print(f"Category {category}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

# Call the function to split the data
split_data()
