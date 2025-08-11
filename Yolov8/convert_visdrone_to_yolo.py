import os
import cv2
import shutil
import zipfile
import yaml
from pathlib import Path
from tqdm import tqdm

def load_config(config_path='config.yaml'):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_dataset(zip_path):
    """
    Extract VisDrone dataset
    Support extracting multiple zip files
    """
    print("Checking if dataset extraction is needed...")
    zip_files = []

    # Find all zip files
    for file in os.listdir(zip_path):
        if file.endswith('.zip') and 'VisDrone' in file:
            zip_files.append(file)

    if not zip_files:
        print("No VisDrone dataset zip files found")
        return

    print(f"Found {len(zip_files)} zip files:")
    for zip_file in zip_files:
        print(f"  - {zip_file}")

    for zip_file in zip_files:
        zip_file_path = os.path.join(zip_path, zip_file)

        # Check if already extracted
        extract_folder_name = zip_file.replace('.zip', '')
        extract_path = os.path.join(zip_path, extract_folder_name)
        print(f"extract_path: {extract_path}")

        if os.path.exists(extract_path):
            print(f"  {extract_folder_name} already exists, skipping extraction")
            continue

        print(f"Extracting {zip_file}...")
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(zip_path)
            print(f"  ✓ {zip_file} extraction completed")
        except Exception as e:
            print(f"  ✗ Failed to extract {zip_file}: {e}")

def convert_visdrone_to_yolo(dataset_path, output_path):
    """
    Convert VisDrone format to YOLO format
    VisDrone format: x,y,width,height,score,class_id,truncation,occlusion
    YOLO format: class_id center_x center_y width height (normalized)
    """

    # VisDrone class mapping to YOLO classes (ignore classes 0 and 11)
    visdrone_to_yolo_class = {
        1: 0,   # pedestrian -> person
        2: 0,   # people -> person (merge as person)
        3: 1,   # bicycle
        4: 2,   # car
        5: 3,   # van
        6: 4,   # truck
        7: 5,   # tricycle
        8: 6,   # awning-tricycle
        9: 7,   # bus
        10: 8,  # motor
    }

    class_names = ['person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

    for split in ['train', 'val', 'test-dev', 'test-challenge']:
        if split == 'train':
            src_folder = os.path.join(dataset_path, 'VisDrone2019-DET-train')
        elif split == 'val':
            src_folder = os.path.join(dataset_path, 'VisDrone2019-DET-val')
        elif split == 'test-dev':
            src_folder = os.path.join(dataset_path, 'VisDrone2019-DET-test-dev')
        else:  # test-challenge
            src_folder = os.path.join(dataset_path, 'VisDrone2019-DET-test-challenge')

        if not os.path.exists(src_folder):
            print(f"Warning: {src_folder} does not exist, skipping...")
            continue

        # Create output directories
        output_images_dir = os.path.join(output_path, 'images', split)
        output_labels_dir = os.path.join(output_path, 'labels', split)
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        images_dir = os.path.join(src_folder, 'images')
        annotations_dir = os.path.join(src_folder, 'annotations')

        if not os.path.exists(images_dir):
            print(f"Warning: {images_dir} does not exist, skipping...")
            continue

        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

        print(f"Processing {split} set: {len(image_files)} images...")

        for img_file in tqdm(image_files):
            # Copy image
            src_img_path = os.path.join(images_dir, img_file)
            dst_img_path = os.path.join(output_images_dir, img_file)
            shutil.copy2(src_img_path, dst_img_path)

            # Read image dimensions
            img = cv2.imread(src_img_path)
            img_height, img_width = img.shape[:2]

            # Process annotation file
            annotation_file = img_file.replace('.jpg', '.txt')
            src_ann_path = os.path.join(annotations_dir, annotation_file)
            dst_ann_path = os.path.join(output_labels_dir, annotation_file)

            yolo_annotations = []

            if os.path.exists(src_ann_path):
                with open(src_ann_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        x, y, w, h, score, class_id = map(int, parts[:6])

                        # Skip ignored regions (0) and others (11)
                        if class_id in visdrone_to_yolo_class:
                            yolo_class = visdrone_to_yolo_class[class_id]

                            # Convert to YOLO format (normalized center coordinates and dimensions)
                            center_x = (x + w / 2) / img_width
                            center_y = (y + h / 2) / img_height
                            norm_width = w / img_width
                            norm_height = h / img_height

                            # Ensure coordinates are within [0,1] range
                            center_x = max(0, min(1, center_x))
                            center_y = max(0, min(1, center_y))
                            norm_width = max(0, min(1, norm_width))
                            norm_height = max(0, min(1, norm_height))

                            yolo_annotations.append(f"{yolo_class} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")

            # Write YOLO format annotation file
            with open(dst_ann_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

    print("Data conversion completed!")
    return class_names

def main():
    """
    Main function: Extract dataset and convert format
    """
    # Load configuration
    config = load_config()

    # Get paths from configuration
    zip_path = "../original_data"  # Assume zip files are in current directory
    output_path = config['dataset_path']

    print(f"Dataset zip path: {zip_path}")
    print(f"Output path: {output_path}")

    # Check and extract dataset
    extract_dataset(zip_path)

    # Convert dataset format
    class_names = convert_visdrone_to_yolo(zip_path, output_path)
    print(f"Class names: {class_names}")

    # Create or update dataset configuration file
    dataset_yaml_path = config['data_config']
    if not os.path.exists(dataset_yaml_path):
        print(f"Creating dataset configuration file: {dataset_yaml_path}")
        dataset_config = {
            'path': output_path,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test-dev',
            'challenge': 'images/test-challenge',
            'nc': len(class_names),
            'names': class_names
        }

        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        print("Dataset configuration file created!")
    else:
        print(f"Dataset configuration file {dataset_yaml_path} already exists")

if __name__ == "__main__":
    main()