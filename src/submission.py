import json
import csv
import os


# -------- CONFIG --------

POSTPROCESSED_PATH = "outputs/predictions/postprocessed.json"
SUBMISSION_PATH = "outputs/submissions/submission.csv"

CONF_THRESHOLD = 0.15   # keep consistent with postprocess


# -------- UTLIS --------
def format_prediction_string(detections):
    """
    Covert detections to Kaggle prediction_string format.
    """

    if not detections:
        return "-"
    
    parts = []

    for det in detections:
        if det["confidence"] < CONF_THRESHOLD:
            continue

        cls_id = det["class"]
        conf = det["confidence"]
        x = det["x"]
        y = det["y"]
        w = det["w"]
        h = det["h"]

        parts.append(
            f"{cls_id} {conf:.4f} {x:.4f} {y:.4f} {w:.4f} {h:.4f}"
        )

    return " ".join(parts) if parts else "-"

# -------- MAIN --------

def main():

    if not os.path.exists(POSTPROCESSED_PATH):
        raise FileNotFoundError("postprocessed.json not found. Run postprocess.py first.")
    
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    with open(POSTPROCESSED_PATH, "r") as f:
        data = json.load(f)

    with open(SUBMISSION_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "right_place", "prediction_string"])

        for item in data:
            filename = item["image"]
            right_place = item["right_place"]
            detections = item["detections"]

            prediction_string = format_prediction_string(detections)

            writer.writerow([
                filename,
                right_place,
                prediction_string
            ])

    print("\nSubmission file created successfully")
    print("Saved to: ", SUBMISSION_PATH)



# -------- ENTRY POINT --------

if __name__ == "__main__":
    main()