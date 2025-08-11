from ultralytics import YOLO
import glob
import os
import re

def find_latest_train_dir():
    """Find the training directory with the highest number"""
    train_dirs = glob.glob("visdrone_detection/train*/")

    if not train_dirs:
        return None

    def extract_number(path):
        match = re.search(r'train(\d+)', path)
        if match:
            return int(match.group(1))
        else:
            return 0

    latest_dir = max(train_dirs, key=extract_number)
    return latest_dir

# Use function to find the latest directory
latest_train_dir = find_latest_train_dir()

if latest_train_dir:
    best_weights = os.path.join(latest_train_dir, "weights", "best.pt")

    print(f"🔍 Found latest training directory: {latest_train_dir}")
    print(f"📊 Using weights file: {best_weights}")

    if os.path.exists(best_weights):
        model = YOLO(best_weights)

        print("🚀 Starting validation and generating images...")

        # Key: specify save location to training directory
        train_name = os.path.basename(latest_train_dir.rstrip('/'))  # Extract names like train6

        results = model.val(
            data="visdrone.yaml",
            plots=True,
            save_json=True,
            conf=0.25,
            iou=0.7,
            project="visdrone_detection",       # ✅ Added: specify project directory
            name=f"{train_name}_validation"     # ✅ Added: specify validation subdirectory name
        )

        print(f"✅ Validation completed! Images saved in: visdrone_detection/{train_name}_validation/")
    else:
        print(f"❌ Weights file does not exist: {best_weights}")
else:
    print("❌ No training directory found")