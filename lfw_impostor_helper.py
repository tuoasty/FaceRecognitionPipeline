import os
import shutil
import random
from pathlib import Path

source_dir = "D:/KEVIN/0SLC/Qualification/Deep Learning/20250704 0915_Deep Learning_Deep Learning QC/Case 5 Dataset/lfw-deepfunneled/lfw-deepfunneled"
dest_dir = "D:/KEVIN/0SLC/RIG/output/preprocessed/probe_negative"

os.makedirs(dest_dir, exist_ok=True)

person_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

print(f"Found {len(person_folders)} person folders in LFW dataset")

random.shuffle(person_folders)
num_impostors = 200
selected_count = 0
counter = 1

for person_folder in person_folders:
    if selected_count >= num_impostors:
        break
    
    person_path = os.path.join(source_dir, person_folder)
    images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if images:
        source_image = os.path.join(person_path, images[0])
        
        ext = os.path.splitext(images[0])[1]

        dest_filename = f"lfw_{counter:03d}{ext}"
        dest_path = os.path.join(dest_dir, dest_filename)

        shutil.copy2(source_image, dest_path)
        
        print(f"Copied {counter}/{num_impostors}: {person_folder} -> {dest_filename}")
        selected_count += 1
        counter += 1

print(f"\nCompleted! Copied {selected_count} impostor images to {dest_dir}")