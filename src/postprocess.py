import json 
import os 
from typing import List, Dict


# --------- CONFIGURATION ---------
RAW_PREDICTIONS_PATH = "outputs/predictions/raw_predictions.json"
POSTPROCESSED_PATH = "outputs/predictions/postprocessed.json"

# Geometry thresholds (tunable)
IOU_THRESHOLD = 0.2      # overlap requiremet 
CONF_THRESHOLD = 0.10    # ignore weak detections


# --------- GEMOTRAY UTILITES ---------

def xywh_to_xyxy(box):
    """
    Convert YOLO normalized xywh -> xyxy
    """

    x, y, w, h = box
    x1 = x-h/2
    y1 = x-w/2
    x2 = x+w/2
    y2 = y+h/2
    return x1, y1, x2, y2


def box_iou(box1, box2):
    """
    Compute IoU between two boxes in xywh (normalized)
    """

    b1 = xywh_to_xyxy(box1)
    b2 = xywh_to_xyxy(box2)

    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    union = area1 + area2 - inter

    if union == 0:
        return 0.0
    
    return inter/union


def center_inside(box_small, box_big):
    """
    Check if center of box_small lies inside box_big
    """
    xs, ys, _, _ = box_small
    xb, yb, wb, hb = box_big


    bx1 = xb-wb/2
    by1 = yb-hb/2
    bx2 = xb+wb/2
    by2 = yb+hb/2 

    return (bx1 <= xs <= bx2) and (by1 <= ys <= by2)


def decide_right_place(detections: List[Dict])->int:
    """
    Decide whether shemagh is worn correctly for one image.
    """

    heads = []
    shemaghs = []

    # Separate detections by class and confidence
    for det in detections:
        if det["confidence"] < CONF_THRESHOLD:
            continue

        if det["class"] == 0:
            heads.append(det)
        elif det["class"]==1:
            shemaghs.append(det)

    # No head or no shemagh -> incorrect 
    if len(heads) == 0 or len(shemaghs) == 0:
        return 0
    
    # Try all head-shemagh pairs 
    for head in heads:
        head_box = (head["x"], head["y"], head["w"], head["h"])

        for sh in shemaghs:
            sh_box = (sh["x"], sh["y"], sh["w"], sh["h"])

            iou = box_iou(head_box, sh_box)

            # Priority 1: shemagh center inside head
            if center_inside(sh_box, head_box):
                return 1
            
            # Priority 2 : any overlap at all
            if box_iou(sh_box, head_box)>0:
                return 1
            
    return 0


# --------  PIPELINE --------

def main():
    if not os.path.exists(RAW_PREDICTIONS_PATH):
        raise FileExistsError("Raw predictins not found. Run infer.py first.")
    
    with open(RAW_PREDICTIONS_PATH, "r") as f:
        data = json.load(f)

    output = []

    correct = 0

    for item in data:
        image_name = item["image"]
        detections = item["detections"]

        right_place = decide_right_place(detections)

        if right_place == 1:
            correct+=1

        output.append({
            "image": image_name,
            "right_place": right_place,
            "detections": detections
        })

    os.makedirs(os.path.dirname(POSTPROCESSED_PATH), exist_ok=True)

    with open(POSTPROCESSED_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("\nPostprocessing completed")
    print("Images processed: ", len(output))
    print("Right place detected: ", correct)
    print("Saved to: ", POSTPROCESSED_PATH)


# ---------- ENTRY POINT ----------

if __name__ == "__main__":
    main()