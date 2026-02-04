import os
import shutil
from sklearn.model_selection import train_test_split


# ==============================
# STEP 0: PATH CONFIGURATION
# ==============================

RAW_DIR = "data/raw/dal_shemagh"

RAW_IMAGES_DIR = os.path.join(RAW_DIR, "images/train")
RAW_LABELS_DIR = os.path.join(RAW_DIR, "labels/train")

OUT_DIR = "data"

TRAIN_IMAGES_DIR = os.path.join(OUT_DIR, "train/images")
TRAIN_LABELS_DIR = os.path.join(OUT_DIR, "train/labels")

VAL_IMAGES_DIR = os.path.join(OUT_DIR, "val/images")
VAL_LABELS_DIR = os.path.join(OUT_DIR, "val/labels")


# ==============================
# STEP 1: VALIDATE DATASET
# ==============================

def validate_structure():

    print("\n[STEP 1] Validating dataset structure...")

    if not os.path.exists(RAW_IMAGES_DIR):
        raise FileNotFoundError("Missing images/train folder")

    if not os.path.exists(RAW_LABELS_DIR):
        raise FileNotFoundError("Missing labels/train folder")

    print("✓ Raw dataset folders found")


# ==============================
# STEP 2: CREATE OUTPUT FOLDERS
# ==============================

def create_output_structure():

    print("\n[STEP 2] Creating output folder structure...")

    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)

    os.makedirs(VAL_IMAGES_DIR, exist_ok=True)
    os.makedirs(VAL_LABELS_DIR, exist_ok=True)

    print("✓ Output folders ready")


# ==============================
# STEP 3: COLLECT IMAGE FILES
# ==============================

def collect_images():

    print("\n[STEP 3] Collecting image files...")

    valid_ext = (".jpg", ".jpeg", ".png")

    images = [
        f for f in os.listdir(RAW_IMAGES_DIR)
        if f.lower().endswith(valid_ext)
    ]

    if len(images) == 0:
        raise RuntimeError("No images found")

    print("✓ Found", len(images), "training images")

    return images


# ==============================
# STEP 4: SPLIT DATASET
# ==============================

def split_dataset(images):

    print("\n[STEP 4] Splitting dataset (80% train / 20% val)...")

    train_imgs, val_imgs = train_test_split(
        images,
        test_size=0.2,
        shuffle=True,
        random_state=42
    )

    print("✓ Train images:", len(train_imgs))
    print("✓ Val images:", len(val_imgs))

    return train_imgs, val_imgs


# ==============================
# STEP 5: COPY FILES
# ==============================

def copy_data(files, target_img_dir, target_lbl_dir, label=""):

    print(f"\n[STEP 5] Copying {label} data...")

    img_count = 0
    lbl_count = 0

    for name in files:

        src_img = os.path.join(RAW_IMAGES_DIR, name)
        dst_img = os.path.join(target_img_dir, name)

        shutil.copy2(src_img, dst_img)
        img_count += 1

        label_name = os.path.splitext(name)[0] + ".txt"
        src_lbl = os.path.join(RAW_LABELS_DIR, label_name)
        dst_lbl = os.path.join(target_lbl_dir, label_name)

        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
            lbl_count += 1

    print(f"✓ Copied {img_count} images and {lbl_count} labels")

    return img_count, lbl_count


# ==============================
# MAIN PIPELINE
# ==============================

def main():

    print("\n========== DATASET PIPELINE STARTED ==========\n")

    validate_structure()
    create_output_structure()

    images = collect_images()

    train_imgs, val_imgs = split_dataset(images)

    copy_data(train_imgs, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, "TRAIN")
    copy_data(val_imgs, VAL_IMAGES_DIR, VAL_LABELS_DIR, "VALIDATION")

    print("\n========== DATASET PIPELINE FINISHED ==========\n")


if __name__ == "__main__":
    main()