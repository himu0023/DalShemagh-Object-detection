import os 
import shutil
from sklearn.model_selection import train_test_split



# ---- PATH CONFIGURATION ----

RAW_DATA_DIR = "../data/raw/dal_shemagh"
PROCESSED_DATA_DIR = "../data"

RAW_IMAGE_DIR = os.path.join(RAW_DATA_DIR, "images")
RAW_LABEL_DIR = os.path.join(RAW_DATA_DIR, "labels")

TRAIN_IMAGE_DIR = os.path.join(PROCESSED_DATA_DIR, "train/images")
TRAIN_LABEL_DIR = os.path.join(PROCESSED_DATA_DIR, "train/labels")

VAL_IMAGE_DIR = os.path.join(PROCESSED_DATA_DIR, "val/images")
VAL_LABEL_DIR = os.path.join(PROCESSED_DATA_DIR, "val/labels")



# ---- UTILITY FUNCTIONS ----

def create_directory_structure():
    """
    Creates required folder structure for YOLO training.

    YOLO expects:
    data/train/images
    data/train/labels
    data/val/images
    data/val/labels
    """

    os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)

    os.makedirs(VAL_IMAGE_DIR, exist_ok=True)
    os.makedirs(VAL_LABEL_DIR, exist_ok=True)


def validate_raw_dataset():
    """
    Checks whether the raw dataset folder exists 
    and contains expected subfolders.

    This avoids silent crashes later.
    """

    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError("Raw dataset folder not found!")
    
    if not os.path.exists(RAW_IMAGE_DIR):
        raise FileNotFoundError("Raw images folder missing!")
    
    if not os.path.exists(RAW_LABEL_DIR):
        raise FileNotFoundError("Raw labels folder missing!")
    

def collect_image_files():
    """
    Collects all valid image filenames from raw dataset.

    Returns:
        List of images filename (not full paths)
    """

    valid_extensions = (".jpg", ".jpeg", ".png")

    image_files = []

    for file_name in os.listdir(RAW_IMAGE_DIR):
        if file_name.lower().endswith(valid_extensions):
            image_files.append(file_name)

    if len(image_files) == 0:
        raise RuntimeError("No images found in raw dataset!")
    
    return image_files


def split_train_validation(image_list, validation_ratio = 0.2):
    """
    Split dataset into training and validationn sets.

    why random_state?
    ----------------
    Ensures reproducibility.
    Same split every time.
    """
    
    train_files, val_files = train_test_split(
        image_list,
        test_size=validation_ratio,
        shuffle=True, 
        random_state=42
    )

    return train_files, val_files



def copy_image_label_pairs(file_list, target_image_dir, target_label_dir):
    """
    Copies image files and their corresponding YOLO label files.

    If label file does not exist:
    - Image will still be copied 
    - Label will be skiped

    This avoids breaking traning when some images
    contain no objects.
    """

    copied_images = 0
    copied_labels = 0

    for file_name in file_list:

        image_src_path = os.path.join(RAW_IMAGE_DIR, file_name)
        image_dst_path = os.path.join(target_label_dir, file_name)


        label_name = file_name.rsplit(".", 1)[0] + ".txt"
        label_src_path = os.path.join(RAW_LABEL_DIR, label_name)
        label_dst_path = os.path.join(target_label_dir, label_name)

        shutil.copy(image_src_path, image_dst_path)
        copied_images+=1

        if os.path.exists(label_src_path):
            shutil.copy(label_src_path, label_dst_path)
            copied_labels+=1

    return copied_images, copied_labels