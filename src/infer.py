import os 
import json 
from ultralytics import YOLO
from tqdm import tqdm

# ---- CONFIGURATION ----

# Path to trained model
MODEL_PATH = "models/shemagh_yolov8n/weights/best.pt"

# Test images directory (Kaggle test set)
TEST_IMAGE_DIR = "data/raw/dal_shemagh/images/test"

# Output folder 
OUTPUT_DIR = "outputs/predictions"


# Inference parameters
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
DEVICE = 0                  # GPU id
BATCH_SIZE = 8              # Safe for RTX 3050


# ---- UTILITY FUNCTIONS ----

def create_output_dir():
    """Create output directory if missing."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_test_images():
    """Collect all test image filenames."""
    
    valid_ext = ('.jpg', '.jpeg', '.png')

    files = [
        f for f in os.listdir(TEST_IMAGE_DIR)
        if f.lower().endswith(valid_ext)
    ]

    if len(files) == 0:
        raise RuntimeError("No test images found!")
    
    print("Total test images:", len(files))

    return files

# ---- INFERENCE PIPELINE ----


def run_inference():
    print("\n-----------")
    print("STARTING INFERENCE")
    print("-----------\n")

    create_output_dir()

    # Load trained model 
    print("Loading trained model....")
    model = YOLO(MODEL_PATH)

    # Load test images list 
    image_files = get_test_images()

    predictions = []

    print("Running inference....\n")

    for img_name in tqdm(image_files):

        img_path = os.path.join(TEST_IMAGE_DIR, img_name)

        # Run YOLO prediction 
        results = model.predict(
            source=img_path,
            imgsz= IMG_SIZE,
            conf = CONF_THRESHOLD, 
            iou = IOU_THRESHOLD,
            device = DEVICE,
            batch = BATCH_SIZE,
            verbose = False
        )

        result = results[0]

        boxes = result.boxes

        image_predictions = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:

                cls_id = int(box.cls.item())
                score = float(box.conf.item())

                # YOLO format: normalized xywh
                x, y, w, h = box.xywh[0].tolist()

                image_predictions.append({
                    "class": cls_id,
                    "confidence": score, 
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                })

        predictions.append({
            "image": img_name,
            "detections": image_predictions
        })

    # Save raw predictions 
    output_path = os.path.join(OUTPUT_DIR, "raw_predictions.json")

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print("\n------------")
    print(" INFERENCE COMPLETED ")
    print("------------")
    print("Predictions saved to: ")
    print(output_path)



# ------ ENTRY POINT ------
if __name__ == "__main__":
    run_inference()