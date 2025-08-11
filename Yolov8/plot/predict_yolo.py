from ultralytics import YOLO
import cv2
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from utils import find_latest_train_dir

def load_config():
    """Load configuration file"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def generate_confidence_heatmap(results, original_image, heatmap_config):
    """
    Generate confidence-based heatmap
    """
    if not heatmap_config.get('enabled', False):
        return None, None

    # Get image dimensions
    h, w = original_image.shape[:2]

    # Create blank heatmap
    heatmap = np.zeros((h, w), dtype=np.float32)

    # Iterate through detection results
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
            confidences = result.boxes.conf.cpu().numpy()  # Get confidences

            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                # Add confidence weights in bounding box area
                heatmap[y1:y2, x1:x2] += conf

    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Apply Gaussian smoothing
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

    # Select colormap
    colormap_name = heatmap_config.get('colormap', 'jet')
    colormap_dict = {
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        'parula': cv2.COLORMAP_PARULA,
        'viridis': cv2.COLORMAP_VIRIDIS
    }
    colormap = colormap_dict.get(colormap_name, cv2.COLORMAP_JET)

    # Convert to colored heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)

    # Create overlay image
    alpha = heatmap_config.get('alpha', 0.6)
    overlay = cv2.addWeighted(original_image, 1-alpha, heatmap_colored, alpha, 0)

    return heatmap_colored, overlay

class YOLOGradCAM:
    """
    Improved Grad-CAM implementation for YOLOv8 models
    Targets feature extraction layers before the detection head
    """
    def __init__(self, model, target_layer_name='model.9'):
        self.model = model.model  # Access the underlying PyTorch model
        self.target_layer = self._get_target_layer(target_layer_name)
        self.gradients = []
        self.activations = []
        self.hooks = []

        # Register hooks
        self._register_hooks()

    def _get_target_layer(self, layer_name):
        """Get target layer by name"""
        try:
            # Handle different layer naming conventions
            if layer_name.startswith('model.'):
                layer_name = layer_name[6:]  # Remove 'model.' prefix

            # Parse layer index
            if layer_name.isdigit():
                layer_idx = int(layer_name)
            elif layer_name == '[-1]' or layer_name == '-1':
                layer_idx = -1
            else:
                layer_idx = int(layer_name)

            # Get the layer from model
            target_layer = self.model.model[layer_idx]
            print(f"Target layer selected: {layer_idx} - {type(target_layer).__name__}")
            return target_layer

        except Exception as e:
            print(f"Warning: Cannot find target layer {layer_name}, using default layer 9")
            return self.model.model[9]  # Use C2f layer as default

    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            # Store activations, ensure it's detached for memory efficiency
            if isinstance(output, torch.Tensor):
                self.activations.append(output.detach())
            elif isinstance(output, (list, tuple)):
                # Handle multi-output layers
                for out in output:
                    if isinstance(out, torch.Tensor):
                        self.activations.append(out.detach())
                        break

        def backward_hook(module, grad_input, grad_output):
            # Store gradients
            if grad_output[0] is not None:
                self.gradients.append(grad_output[0].detach())

        # Register hooks and store handles for cleanup
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

        self.hooks.extend([forward_handle, backward_handle])

    def generate_cam(self, input_tensor, target_category=None):
        """
        Generate Class Activation Map for YOLOv8
        """
        # Clear previous stored values
        self.gradients.clear()
        self.activations.clear()

        # Ensure model is in training mode for gradient computation
        self.model.train()

        # Clear previous gradients
        self.model.zero_grad()

        # Forward pass with gradient computation enabled
        with torch.enable_grad():
            # Run inference and get raw model output
            output = self.model(input_tensor)

            # YOLOv8 returns list of tensors from different scales
            # We need to process the output to get a scalar for backprop
            if isinstance(output, (list, tuple)):
                # Take the first output (usually the main prediction)
                main_output = output[0]
            else:
                main_output = output

            # Extract confidence scores and compute a target for backprop
            if isinstance(main_output, torch.Tensor) and main_output.dim() >= 2:
                # For YOLOv8, the output format is typically [batch, anchors, predictions]
                # Where predictions = [x, y, w, h, confidence, class_probs...]

                # Find predictions with confidence > threshold
                if main_output.shape[-1] > 4:  # Has confidence scores
                    conf_scores = main_output[..., 4]  # Extract confidence scores

                    # Use top-k confidences for more stable gradients
                    top_k = min(10, conf_scores.numel())
                    top_confs, _ = torch.topk(conf_scores.flatten(), top_k)
                    target_score = torch.mean(top_confs)

                    print(f"Target score for backprop: {target_score.item():.4f}")

                    # Backpropagate from the target score
                    if target_score.requires_grad:
                        target_score.backward(retain_graph=True)
                    else:
                        print("Warning: Target score has no gradients")
                        return None
                else:
                    print("Warning: Unexpected output format - no confidence scores found")
                    return None
            else:
                print("Warning: Unexpected output tensor format")
                return None

        # Restore model to eval mode
        self.model.eval()

        # Check if we captured gradients and activations
        if not self.gradients or not self.activations:
            print("Warning: Could not obtain gradients or activations")
            print(f"Gradients captured: {len(self.gradients)}")
            print(f"Activations captured: {len(self.activations)}")
            return None

        # Use the most recent gradients and activations
        gradients = self.gradients[-1]
        activations = self.activations[-1]

        print(f"Gradients shape: {gradients.shape}")
        print(f"Activations shape: {activations.shape}")

        try:
            # Ensure both tensors have the same shape
            if gradients.shape != activations.shape:
                print("Warning: Shape mismatch between gradients and activations")
                # Try to match dimensions
                min_dim = min(gradients.dim(), activations.dim())
                if min_dim >= 3:
                    # Keep the spatial dimensions and adjust channels if needed
                    if gradients.dim() == 4 and activations.dim() == 4:
                        min_channels = min(gradients.shape[1], activations.shape[1])
                        gradients = gradients[:, :min_channels]
                        activations = activations[:, :min_channels]
                    else:
                        print("Cannot handle dimension mismatch")
                        return None
                else:
                    print("Insufficient dimensions for CAM computation")
                    return None

            # Compute importance weights (global average pooling over spatial dimensions)
            if gradients.dim() == 4:  # [B, C, H, W]
                weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
                # Compute weighted feature maps
                cam = torch.sum(weights * activations, dim=1, keepdim=True)
            elif gradients.dim() == 3:  # [B, C, HW] or similar
                weights = torch.mean(gradients, dim=2, keepdim=True)
                cam = torch.sum(weights * activations, dim=1, keepdim=True)
            else:
                print(f"Unsupported tensor dimensions: {gradients.dim()}")
                return None

            # Apply ReLU activation to focus on positive contributions
            cam = F.relu(cam)

            # Normalize to [0,1]
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                print("Warning: CAM has no variation")
                return None

            return cam

        except Exception as e:
            print(f"CAM computation error: {e}")
            return None

    def cleanup(self):
        """Remove hooks to prevent memory leaks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

def generate_attention_map(model, image_path, attention_config):
    """
    Generate attention map using Grad-CAM
    """
    if not attention_config.get('enabled', False):
        return None, None

    try:
        # Read original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Error: Cannot read image {image_path}")
            return None, None

        # Preprocess image for YOLO
        img_size = 1280  # Use image size from config

        # Maintain aspect ratio scaling
        h, w = original_image.shape[:2]
        scale = min(img_size / h, img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize image
        resized_img = cv2.resize(original_image, (new_w, new_h))

        # Create padded image
        padded_img = np.full((img_size, img_size, 3), 114, dtype=np.uint8)

        # Place resized image in center of padded image
        y_offset = (img_size - new_h) // 2
        x_offset = (img_size - new_w) // 2
        padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

        # Convert to tensor and normalize
        input_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Move to appropriate device
        device = next(model.model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)

        # Create Grad-CAM object with a better target layer
        target_layer = attention_config.get('target_layer', 'model.9')
        print(f"Using target layer: {target_layer}")

        grad_cam = YOLOGradCAM(model, target_layer)

        # Generate CAM
        cam = grad_cam.generate_cam(input_tensor)

        # Cleanup hooks
        grad_cam.cleanup()

        if cam is None:
            print("Warning: Failed to generate attention map")
            return None, None

        # Convert to numpy array
        cam_np = cam.squeeze().cpu().detach().numpy()

        # Resize to original image size
        upsampling = attention_config.get('upsampling', 'bilinear')

        if upsampling == 'bilinear':
            cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_NEAREST)

        # Select colormap
        colormap_name = attention_config.get('colormap', 'jet')
        colormap_dict = {
            'jet': cv2.COLORMAP_JET,
            'hot': cv2.COLORMAP_HOT,
            'parula': cv2.COLORMAP_PARULA,
            'viridis': cv2.COLORMAP_VIRIDIS
        }
        colormap = colormap_dict.get(colormap_name, cv2.COLORMAP_JET)

        # Convert to colored attention map
        attention_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)

        # Create overlay image
        alpha = attention_config.get('alpha', 0.6)
        overlay = cv2.addWeighted(original_image, 1-alpha, attention_colored, alpha, 0)

        return attention_colored, overlay

    except Exception as e:
        print(f"Error generating attention map: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def predict_single_image(model, image_path, pred_config, device):
    """
    Predict on a single image and return results
    """
    results = model.predict(
        source=image_path,
        imgsz=pred_config['imgsz'],
        conf=pred_config['conf'],
        iou=pred_config['iou'],
        device=device,
        save=False,
        verbose=False
    )
    return results

def process_single_image(model, image_path, config):
    """
    Process a single image with all visualizations
    """
    pred_config = config['prediction']
    heatmap_config = config.get('heatmap', {})
    attention_config = config.get('attention_map', {})
    device = config.get('device', 'mps')

    print(f"Processing: {os.path.basename(image_path)}")

    # Read original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Cannot read image {image_path}")
        return

    # Get predictions for visualizations
    results = predict_single_image(model, image_path, pred_config, device)
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # Generate confidence heatmap
    if heatmap_config.get('enabled', False):
        print("Generating confidence heatmap...")
        heatmap_colored, heatmap_overlay = generate_confidence_heatmap(results, original_image, heatmap_config)

        if heatmap_colored is not None:
            output_dir = "runs/detect/predict_heatmap"
            os.makedirs(output_dir, exist_ok=True)

            if heatmap_config.get('save_heatmap', True):
                heatmap_path = os.path.join(output_dir, f"{filename}_heatmap.jpg")
                cv2.imwrite(heatmap_path, heatmap_colored)
                print(f"Heatmap saved: {heatmap_path}")

            if heatmap_config.get('save_overlay', True):
                overlay_path = os.path.join(output_dir, f"{filename}_heatmap_overlay.jpg")
                cv2.imwrite(overlay_path, heatmap_overlay)
                print(f"Heatmap overlay saved: {overlay_path}")

    # Generate attention map
    if attention_config.get('enabled', False):
        print("Generating attention map...")
        attention_colored, attention_overlay = generate_attention_map(model, image_path, attention_config)

        if attention_colored is not None:
            output_dir = "runs/detect/predict_attention"
            os.makedirs(output_dir, exist_ok=True)

            if attention_config.get('save_attention', True):
                attention_path = os.path.join(output_dir, f"{filename}_attention.jpg")
                cv2.imwrite(attention_path, attention_colored)
                print(f"Attention map saved: {attention_path}")

            if attention_config.get('save_overlay', True):
                overlay_path = os.path.join(output_dir, f"{filename}_attention_overlay.jpg")
                cv2.imwrite(overlay_path, attention_overlay)
                print(f"Attention overlay saved: {overlay_path}")

def process_directory(model, source_dir, config):
    """
    Process all images in a directory
    """
    print("Processing images in directory...")

    # Get image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(image_extensions)]

    if not image_files:
        print("No image files found in directory!")
        return

    print(f"Found {len(image_files)} image files")

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(source_dir, img_file)
        print(f"Processing {i+1}/{len(image_files)}: {img_file}")

        try:
            process_single_image(model, img_path, config)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

def inspect_model_layers(model):
    """
    Inspect model layers to find suitable targets for Grad-CAM
    """
    print("\nModel architecture inspection:")
    print("=" * 50)

    for i, layer in enumerate(model.model.model):
        print(f"Layer {i}: {type(layer).__name__}")
        if hasattr(layer, 'cv1') or hasattr(layer, 'cv2') or hasattr(layer, 'cv3'):
            print(f"  -> Suitable for Grad-CAM (has convolution layers)")

    print("=" * 50)
    print("Recommended target layers:")
    print("  - model.9: Last C2f layer before detection head")
    print("  - model.8: Second to last C2f layer")
    print("  - model.7: Earlier feature extraction layer")
    print()

def main():
    """
    Main function for inference using trained model
    """
    print("YOLOv8 Inference with Visualization")
    print("=" * 50)

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    pred_config = config['prediction']
    heatmap_config = config.get('heatmap', {})
    attention_config = config.get('attention_map', {})

    train_dir=find_latest_train_dir(config)
    print(f"Latest train directory: {train_dir}")

    weights_path = f"{train_dir}/weights/best.pt"

    if not os.path.exists(weights_path):
        print(f"Error: Weights file {weights_path} does not exist!")
        return

    try:
        model = YOLO(weights_path)
        print(f"Model loaded successfully from {weights_path}")

        # Inspect model layers for debugging
        if attention_config.get('enabled', False):
            inspect_model_layers(model)

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Select inference source
    source = input("Enter image path or directory (e.g., bus.jpg or VisDrone2019-DET-test-challenge/images): ").strip()

    if not os.path.exists(source):
        print(f"Error: Path {source} does not exist!")
        return

    # Perform standard inference first
    print("Running standard inference...")
    try:
        results = model.predict(
            source=source,
            imgsz=pred_config['imgsz'],
            conf=pred_config['conf'],
            iou=pred_config['iou'],
            device=config.get('device', 'mps'),
            save=True,
            save_txt=True,
            save_conf=True,
            show_labels=True,
            show_conf=True,
        )
        print(f"Standard inference completed! Results saved in: runs/detect/predict/")
        print(f"Configuration used: imgsz={pred_config['imgsz']}, conf={pred_config['conf']}, iou={pred_config['iou']}")
    except Exception as e:
        print(f"Error during standard inference: {e}")
        return

    # Process visualizations
    try:
        if os.path.isfile(source) and source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            # Process single image
            process_single_image(model, source, config)

            # Show results for single image
            if not heatmap_config.get('enabled', False) and not attention_config.get('enabled', False):
                result = results[0]
                result.show()

        elif os.path.isdir(source):
            # Process directory
            process_directory(model, source, config)
        else:
            print("Invalid source: must be an image file or directory")

    except Exception as e:
        print(f"Error during visualization processing: {e}")
        import traceback
        traceback.print_exc()

    print("Processing completed!")

if __name__ == "__main__":
    main()