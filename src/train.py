print("train.py file started")


import os 
from ultralytics import YOLO

# ---- CONFIGURATION SECTION ----

# Path to dataset YAML
DATA_YAML = "configs/shemagh.yaml"

# Pretrained YOLOv8 model
MODEL_NAME = "yolov8s.pt" # nano model (fast + fits RTX 3050)

# Output directory for trained models
OUTPUT_DIR = "models"

# Training hyperparameters 
EPOCHS = 80         # Increase later (start with 30-50)
IMG_SIZE = 640
BATCH_SIZE = 8      # Safe for RTX 3050 6GB
WORKERS = 2         # Prevent Windows memory crash
DEVICE = 0          # GPU id (0 = first GPU)

# Experiment name
EXPERIMENT_NAME = "shemagh_yolov8n"


# ---- TRAINING FUNCTION ----

def train():

    print("\n-------")
    print(" SHEMAH YOLO TRAINIG START ")
    print("-------\n")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)

    print("Starting training...")

    results = model.train(
        data = DATA_YAML,
        epochs = EPOCHS,
        imgsz = IMG_SIZE,
        batch = BATCH_SIZE,
        workers = WORKERS,
        device = DEVICE,

        # Output Control 
        project = OUTPUT_DIR, 
        name = EXPERIMENT_NAME,
        exist_ok = True,

        # Performance and stability

        amp = True,         # Mixed precision
        cache = False,      # Avoid RAM overflow on Windows
        patience = 10,      # Early stopping patience

        # Logging 
        verbose = True
    )
        
    

    print("Best model saved at: ")
    print(results.save_dir)


    # ---- Entry Point ----

if __name__ == "__main__":
    train()