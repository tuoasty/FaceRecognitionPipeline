import os
import cv2
import shutil

folder = "D:/KEVIN/0SLC/RIG/output/preprocessed/probe_labeled_v0/negative"

# Create output folders
true_impostors_folder = os.path.join(folder, "true_impostors")
corrected_folder = os.path.join(folder, "corrected")

os.makedirs(true_impostors_folder, exist_ok=True)
os.makedirs(corrected_folder, exist_ok=True)

files = [f for f in os.listdir(folder) 
         if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for f in files:
    path = os.path.join(folder, f)
    img = cv2.imread(path)
    
    if img is None:
        print(f"⚠ Could not read: {f}")
        continue

    cv2.imshow("Image", img)
    print("Current filename:", f)
    
    # Wait for the window to render
    cv2.waitKey(1)
    
    new_prefix = input("Enter new label/prefix (empty to skip): ").strip()
    
    if new_prefix:
        # Rename and move to corrected folder
        parts = f.split("_")
        parts[0] = new_prefix
        new_name = "_".join(parts)
        new_path = os.path.join(corrected_folder, new_name)
        shutil.move(path, new_path)
        print(f"✓ Renamed → {new_name} (moved to corrected/)")
    else:
        # Move to true impostors folder
        new_path = os.path.join(true_impostors_folder, f)
        shutil.move(path, new_path)
        print(f"✓ Skipped (moved to true_impostors/)")

    cv2.destroyAllWindows()

cv2.destroyAllWindows()
print("\n✓ Processing complete!")
print(f"True impostors: {true_impostors_folder}")
print(f"Corrected: {corrected_folder}")