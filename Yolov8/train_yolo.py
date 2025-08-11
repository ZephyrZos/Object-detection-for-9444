import os
import subprocess
import sys
import time
# Must set these environment variables before importing any libraries that use MKL!
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from ultralytics import YOLO
import torch
import yaml
from utils import load_config, get_device, get_cpu_optimized_params, print_training_info, find_latest_weights

def sync_and_shutdown():
    """Sync files and shutdown after training completion"""
    try:
        print("\n" + "="*50)
        print("🔄 Starting file sync to mounted directory...")

        # Execute rsync command
        rsync_cmd = [
            "rsync",
            "-a",
            "/root/project/",  # Note: trailing slash in source directory is important
            "/root/private/project/",
            "--info=progress2"
        ]

        print(f"Executing command: {' '.join(rsync_cmd)}")
        result = subprocess.run(rsync_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ File sync successful!")
            print(result.stdout)
        else:
            print("❌ File sync failed:")
            print(result.stderr)
            return False

        # print("\n💾 Sync complete, preparing to shutdown...")
        # time.sleep(5)  # Wait 5 seconds to ensure all operations complete

        # Shutdown command
        # print("🔌 Shutting down...")
        # subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)

    except Exception as e:
        print(f"❌ Error during sync or shutdown process: {e}")
        print("🔧 Please manually execute sync and shutdown operations")
        return False

    return True

# These functions have been moved to utils.py

def main():
    """
    Train VisDrone dataset using YOLO
    """
    try:
        # Load configuration
        config = load_config()
        dataset_path = config['dataset_path']
        model_pretrained = config['model_pretrained']

        # Check if dataset exists
        if not os.path.exists(dataset_path):
            print(f"Error: {dataset_path} directory does not exist!")
            print("Please run 'python convert_visdrone_to_yolo.py' to convert dataset first")
            return

        # Auto detect device
        device = get_device(config['device'])
        print(f"Using device: {device}")

        # If CPU, adjust specific parameters
        if device == 'cpu':
            config = get_cpu_optimized_params(config)

        # Load pretrained model or resume from checkpoint
        weights_path = config['model_pretrained']

        # Check for existing training weights
        existing_weight = find_latest_weights(config)
        if existing_weight:
            print(f"🔄 Found existing training weights: {existing_weight}")
            response = input("Resume training from checkpoint? (Y/n): ").strip().lower()
            if response in ['', 'y', 'yes']:
                weights_path = existing_weight
                resume_training = True
                print(f"🔄 Resuming training from: {weights_path}")
            else:
                print(f"🆕 Starting from pretrained model: {weights_path}")
                resume_training = False
        else:
            print(f"🆕 Starting from pretrained model: {weights_path}")
            resume_training = False

        print(f"Loading model: {weights_path}")

        model = YOLO(weights_path)

        # Extract training and augmentation parameters from config
        training_params = config['training']
        aug_params = config['augmentation']

        # Merge dictionaries to pass to model.train()
        train_args = {**training_params, **aug_params}

        # Display detailed training information
        print_training_info(config, device)

        # Start training
        print("🚀 Starting training...")
        results = model.train(
            data=config['data_config'],
            device=device,
            # Use runs/detect if project_name doesn't exist in config file
            project=config.get('project_name', 'runs/detect'),
            name='train_resume' if resume_training else 'train',
            resume=resume_training,
            seed=config.get('seed', 0),
            # Unpack all parameters from config dictionary
            **train_args
        )

        print("🎉 Training completed!")
        print(f"Best weights saved at: {results.save_dir}/weights/best.pt")
        print(f"Last weights saved at: {results.save_dir}/weights/last.pt")

        # Generate loss charts
        results_csv = os.path.join(results.save_dir, 'results.csv')
        if os.path.exists(results_csv):
            print("📊 Generating training charts...")
            from utils import plot_training_results
            success = plot_training_results(results_csv)
            if success:
                print(f"✅ Training charts generated: {results.save_dir}/results.png")
            else:
                print("⚠️ Chart generation failed, but training data has been saved")
        else:
            print("⚠️ results.csv file not found, unable to generate charts")

        # Auto sync and shutdown after training completion
        sync_and_shutdown()

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during training process: {e}")
        sync_and_shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()