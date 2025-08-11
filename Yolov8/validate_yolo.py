from ultralytics import YOLO
import os
import yaml
import torch
import glob

def load_config(config_path='config.yaml'):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_device(config_device):
    """Auto detect the most suitable device"""
    if config_device != 'auto':
        return config_device
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def find_latest_weights(config):
    """Auto find latest training weights"""
    project_name = config.get('project_name', 'runs/detect')

    # Find all possible weight files
    weight_patterns = [
        f"{project_name}/train*/weights/best.pt",
        f"{project_name}/train*/weights/last.pt",
        "runs/detect/train*/weights/best.pt",  # Backup path
        "runs/detect/train*/weights/last.pt"   # Backup path
    ]

    all_weights = []
    for pattern in weight_patterns:
        all_weights.extend(glob.glob(pattern))

    if all_weights:
        # Sort by modification time, return the latest
        latest_weight = max(all_weights, key=os.path.getmtime)
        return latest_weight

    return None

def main():
    """
    Validate trained YOLO model
    """
    # Load configuration
    config = load_config()
    val_config = config['validation']
    dataset_config = config['data_config']
    project_name = config.get('project_name', 'runs/detect')

    # Auto find latest weights
    latest_weight = find_latest_weights(config)

    if latest_weight:
        print(f"🔍 Found latest weight file: {latest_weight}")
        use_latest = input("Use this weight file? (Y/n): ").strip().lower()
        if use_latest in ['', 'y', 'yes']:
            weights_path = latest_weight
        else:
            weights_path = input(f"Please enter weight file path (e.g.: {project_name}/train/weights/best.pt): ").strip()
    else:
        weights_path = input(f"请输入权重文件路径 (例: {project_name}/train/weights/best.pt): ").strip()

    if not os.path.exists(weights_path):
        print(f"Error: Weight file {weights_path} does not exist!")
        return
    weight_dir = os.path.dirname(os.path.dirname(weights_path))  # Remove /weights/best.pt
    train_name = os.path.basename(weight_dir)

    model = YOLO(weights_path)

    # Auto detect device
    device = get_device(config['device'])
    print(f"Using device: {device}")

    # Validate on validation set
    print("Validating on validation set...")
    val_results = model.val(
        data=dataset_config,
        device=device,
        project=project_name,
        name=f"{train_name}_validation",
        **val_config
    )

    print(f"Validation results:")
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    print(f"📊 Validation results (including confusion matrix) saved at: {project_name}/{train_name}_validation/")

    # Test on test set (if available)
    test_set_path = os.path.join(config['dataset_path'], 'images', 'test-dev')
    if os.path.exists(test_set_path):
        print("\nTesting on test set...")
        # Note: Test set validation also uses validation set parameter configuration
        test_results = model.val(
            data=dataset_config,
            split='test',
            device=device,
            **val_config
        )
        print(f"Test results:")
        print(f"mAP50: {test_results.box.map50:.4f}")
        print(f"mAP50-95: {test_results.box.map:.4f}")

if __name__ == "__main__":
    main()