import os
import shutil
from tqdm import tqdm

base_fisheye_path = r"DataSets\\Fisheye8K_all_including_train&test"
train_img_dir = os.path.join(base_fisheye_path, "train", "images")
train_lbl_dir = os.path.join(base_fisheye_path, "train", "labels")
val_img_dir = os.path.join(base_fisheye_path, "test", "images")
val_lbl_dir = os.path.join(base_fisheye_path, "test", "labels")


woodscape_img_dir = os.path.join(base_fisheye_path, "Woodscape", "images")
woodscape_lbl_dir = os.path.join(base_fisheye_path, "Woodscape", "labels")

merged_img_dir = os.path.join(base_fisheye_path, "train_test_merged", "images")
merged_lbl_dir = os.path.join(base_fisheye_path, "train_test_merged", "labels")

def ensure_dir(path):
    """Create directory if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)

def copy_with_prefix(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, prefix):
    """Copy all image and label files from source to destination with prefix."""
    img_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.png'))]
    for img_file in tqdm(img_files, desc=f"Merging {prefix}"):
        name, ext = os.path.splitext(img_file)
        img_src = os.path.join(src_img_dir, img_file)
        lbl_src = os.path.join(src_lbl_dir, f"{name}.txt")

        new_img_name = f"{prefix}_{name}{ext}"
        new_lbl_name = f"{prefix}_{name}.txt"

        img_dst = os.path.join(dst_img_dir, new_img_name)
        lbl_dst = os.path.join(dst_lbl_dir, new_lbl_name)

        shutil.copy2(img_src, img_dst)

        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, lbl_dst)

print("Creating output directories...")
ensure_dir(merged_img_dir)
ensure_dir(merged_lbl_dir)

print("\nStarting dataset merge...")

copy_with_prefix(train_img_dir, train_lbl_dir, merged_img_dir, merged_lbl_dir, "fisheye_train")
copy_with_prefix(val_img_dir, val_lbl_dir, merged_img_dir, merged_lbl_dir, "fisheye_test")
copy_with_prefix(woodscape_img_dir, woodscape_lbl_dir, merged_img_dir, merged_lbl_dir, "woodscape")

print("\nDataset merging completed successfully!")
print(f"Merged images saved in: {merged_img_dir}")
print(f"Merged labels saved in: {merged_lbl_dir}")
