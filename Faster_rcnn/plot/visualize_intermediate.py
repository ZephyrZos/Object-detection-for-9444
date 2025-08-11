#!/usr/bin/env python3
"""
Visualize intermediate results from Faster R-CNN model
Including feature maps, attention maps, and confidence heatmaps
"""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import load_pretrained_model
from utils import get_device
from dataset import get_transform


class IntermediateVisualizer:
    """Visualizer for intermediate results"""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.feature_maps = {}
        self.hooks = []

    def register_hooks(self):
        """Register hooks to capture intermediate features"""

        # Hook for backbone features at different levels
        def hook_fn(name):
            def hook(module, input, output):
                self.feature_maps[name] = output
            return hook

        # Register hooks for FPN layers
        if hasattr(self.model.model, 'backbone'):
            backbone = self.model.model.backbone

            # Hook for different FPN levels
            if hasattr(backbone, 'fpn'):
                # FPN output features
                self.hooks.append(
                    backbone.fpn.register_forward_hook(hook_fn('fpn_features'))
                )

            # Hook for backbone body output
            if hasattr(backbone, 'body'):
                # ResNet layers
                if hasattr(backbone.body, 'layer1'):
                    self.hooks.append(
                        backbone.body.layer1.register_forward_hook(hook_fn('layer1'))
                    )
                if hasattr(backbone.body, 'layer2'):
                    self.hooks.append(
                        backbone.body.layer2.register_forward_hook(hook_fn('layer2'))
                    )
                if hasattr(backbone.body, 'layer3'):
                    self.hooks.append(
                        backbone.body.layer3.register_forward_hook(hook_fn('layer3'))
                    )
                if hasattr(backbone.body, 'layer4'):
                    self.hooks.append(
                        backbone.body.layer4.register_forward_hook(hook_fn('layer4'))
                    )

        # Hook for RPN features
        if hasattr(self.model.model, 'rpn'):
            rpn = self.model.model.rpn
            if hasattr(rpn, 'head'):
                self.hooks.append(
                    rpn.head.cls_logits.register_forward_hook(hook_fn('rpn_cls_logits'))
                )

    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def visualize_feature_maps(self, image_path, output_dir):
        """Visualize feature maps from different layers"""
        print("Generating feature maps visualization...")

        # Load and preprocess image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = get_transform(train=False, config=self.config)
        image_tensor = transform(image_rgb).unsqueeze(0).to(self.device)

        # Register hooks
        self.register_hooks()

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image_tensor)

        # Create output directory
        feat_dir = Path(output_dir) / "feature_maps"
        feat_dir.mkdir(parents=True, exist_ok=True)

        # Visualize feature maps for each layer
        for layer_name, features in self.feature_maps.items():
            if isinstance(features, dict):
                # FPN features are dictionary
                for level, feat in features.items():
                    self._visualize_single_feature_map(
                        feat, f"{layer_name}_{level}", feat_dir, image_rgb
                    )
            else:
                self._visualize_single_feature_map(
                    features, layer_name, feat_dir, image_rgb
                )

        # Remove hooks
        self.remove_hooks()

        print(f"Feature maps saved to: {feat_dir}")

    def _visualize_single_feature_map(self, features, name, output_dir, original_image):
        """Visualize a single feature map"""
        # Get the feature map
        if features.dim() == 4:  # batch_size, channels, height, width
            feat = features[0]  # Remove batch dimension
        else:
            feat = features

        # Select top N channels with highest activation
        n_channels = min(16, feat.shape[0])

        # Calculate mean activation per channel
        channel_means = feat.mean(dim=(1, 2))
        top_channels = torch.argsort(channel_means, descending=True)[:n_channels]

        # Create figure
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.ravel()

        # Plot original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Plot feature maps
        for idx, channel_idx in enumerate(top_channels):
            ax = axes[idx + 1]

            # Get single channel
            feat_map = feat[channel_idx].cpu().numpy()

            # Normalize for visualization
            feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)

            # Resize to original image size for overlay
            feat_map_resized = cv2.resize(
                feat_map, (original_image.shape[1], original_image.shape[0])
            )

            # Create colored heatmap
            im = ax.imshow(feat_map_resized, cmap='jet', alpha=0.7)
            ax.imshow(original_image, alpha=0.3)
            ax.set_title(f"Channel {channel_idx.item()}")
            ax.axis('off')

        # Remove empty subplots
        for idx in range(n_channels + 1, 20):
            axes[idx].axis('off')

        plt.suptitle(f"Feature Maps: {name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_features.png", dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_rpn_attention(self, image_path, output_dir):
        """Visualize RPN attention/objectness scores"""
        print("Generating RPN attention visualization...")

        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = get_transform(train=False, config=self.config)
        image_tensor = transform(image_rgb).unsqueeze(0).to(self.device)

        # Get RPN outputs
        self.model.eval()
        with torch.no_grad():
            # Access internal RPN outputs
            images, _ = self.model.model.transform(image_tensor, None)
            features = self.model.model.backbone(images.tensors)

            # Convert features dict to list format expected by RPN
            if isinstance(features, dict):
                # FPN returns OrderedDict, convert to list
                features_list = list(features.values())
            else:
                features_list = features

            # Get objectness scores from RPN
            objectness, _ = self.model.model.rpn.head(features_list)

        # Create output directory
        rpn_dir = Path(output_dir) / "rpn_attention"
        rpn_dir.mkdir(parents=True, exist_ok=True)

        # Process objectness scores for each feature level
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        # Original image
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Visualize objectness at different scales
        for idx, obj_scores in enumerate(objectness):
            if idx >= 5:  # Limit to 5 feature levels
                break

            # Convert logits to probabilities
            obj_probs = torch.sigmoid(obj_scores[0])

            # Average across anchor types
            obj_map = obj_probs.mean(dim=0).cpu().numpy()

            # Resize to original image size
            obj_map_resized = cv2.resize(
                obj_map, (image_rgb.shape[1], image_rgb.shape[0])
            )

            # Plot
            ax = axes[idx + 1]
            im = ax.imshow(obj_map_resized, cmap='hot', alpha=0.7)
            ax.imshow(image_rgb, alpha=0.3)
            ax.set_title(f"RPN Objectness - Level {idx}")
            ax.axis('off')

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle("RPN Objectness Scores (Attention)", fontsize=16)
        plt.tight_layout()
        plt.savefig(rpn_dir / "rpn_attention.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"RPN attention saved to: {rpn_dir}")

    def visualize_detection_heatmap(self, image_path, output_dir):
        """Visualize detection confidence as heatmap"""
        print("Generating detection confidence heatmap...")

        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        transform = get_transform(train=False, config=self.config)
        image_tensor = transform(image_rgb).unsqueeze(0).to(self.device)

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Create output directory
        heatmap_dir = Path(output_dir) / "confidence_heatmap"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        # Create confidence heatmap
        confidence_map = np.zeros((h, w))

        # Get predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        # Apply confidence threshold
        threshold = self.config['prediction']['confidence_threshold']
        mask = scores > threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # Create per-class heatmaps
        class_names = list(self.config['classes'].values())
        unique_labels = np.unique(labels)

        n_classes = len(unique_labels)
        n_cols = 3
        n_rows = (n_classes + 2) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.ravel()

        # Overall heatmap
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Add gaussian-like confidence distribution
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            for y in range(y1, y2):
                for x in range(x1, x2):
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    max_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2
                    weight = np.exp(-dist**2 / (2 * max_dist**2)) * score
                    confidence_map[y, x] = max(confidence_map[y, x], weight)

        # Plot overall heatmap
        ax = axes[0]
        im = ax.imshow(confidence_map, cmap='jet', alpha=0.7)
        ax.imshow(image_rgb, alpha=0.3)
        ax.set_title("Overall Confidence Heatmap")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Per-class heatmaps
        for idx, class_id in enumerate(unique_labels):
            if idx + 1 >= len(axes):
                break

            ax = axes[idx + 1]

            # Create class-specific heatmap
            class_map = np.zeros((h, w))
            class_boxes = boxes[labels == class_id]
            class_scores = scores[labels == class_id]

            for box, score in zip(class_boxes, class_scores):
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                        max_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2
                        weight = np.exp(-dist**2 / (2 * max_dist**2)) * score
                        class_map[y, x] = max(class_map[y, x], weight)

            im = ax.imshow(class_map, cmap='jet', alpha=0.7)
            ax.imshow(image_rgb, alpha=0.3)

            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            ax.set_title(f"{class_name} Confidence")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused axes
        for idx in range(len(unique_labels) + 1, len(axes)):
            axes[idx].axis('off')

        plt.suptitle("Detection Confidence Heatmaps", fontsize=16)
        plt.tight_layout()
        plt.savefig(heatmap_dir / "confidence_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Confidence heatmap saved to: {heatmap_dir}")

    def visualize_anchor_boxes(self, image_path, output_dir):
        """Visualize anchor boxes at different scales"""
        print("Generating anchor boxes visualization...")

        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = get_transform(train=False, config=self.config)
        image_tensor = transform(image_rgb).unsqueeze(0).to(self.device)

        # Get image size after transformation
        self.model.eval()
        with torch.no_grad():
            images, _ = self.model.model.transform(image_tensor, None)
            features = self.model.model.backbone(images.tensors)

        # Get anchor boxes
        anchor_generator = self.model.model.rpn.anchor_generator
        image_sizes = images.image_sizes

        # Get feature map sizes
        feature_map_sizes = []
        if isinstance(features, dict):
            for feature_name, feature in features.items():
                feature_map_sizes.append(feature.shape[-2:])
            features_list = list(features.values())
        else:
            for feature in features:
                feature_map_sizes.append(feature.shape[-2:])
            features_list = features

        # Generate anchors - pass image list and feature list
        anchors = anchor_generator(images, features_list)

        # Create output directory
        anchor_dir = Path(output_dir) / "anchor_boxes"
        anchor_dir.mkdir(parents=True, exist_ok=True)

        # Get anchors for the first image
        anchors_per_image = anchors[0]  # This is a tensor of all anchors for first image

        # Debug: print anchor structure
        print(f"Anchors shape: {anchors_per_image.shape}")
        print(f"Number of feature levels: {len(features_list)}")

        # Visualize anchors
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        # Original image
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Sample and visualize anchors from all levels combined
        # Since anchors are combined from all feature levels, we'll sample from the entire set
        n_anchors = len(anchors_per_image)
        sample_indices = np.random.choice(n_anchors, min(200, n_anchors), replace=False)

        # Create visualization showing different density of anchors
        anchor_densities = [50, 100, 200, 300, 500]  # Different numbers of anchors to show

        for plot_idx, n_sample in enumerate(anchor_densities):
            if plot_idx + 1 >= len(axes):
                break

            ax = axes[plot_idx + 1]
            ax.imshow(image_rgb, alpha=0.5)

            # Sample anchors for this plot
            current_sample = min(n_sample, n_anchors)
            current_indices = np.random.choice(n_anchors, current_sample, replace=False)

            # Draw sampled anchors
            for anchor_idx in current_indices:
                anchor_box = anchors_per_image[anchor_idx].cpu().numpy()

                # Check if anchor_box has the right shape
                if anchor_box.shape == (4,):
                    x1, y1, x2, y2 = anchor_box
                else:
                    print(f"Unexpected anchor shape: {anchor_box.shape}")
                    continue

                # Scale back to original image size
                scale_x = image_rgb.shape[1] / images.image_sizes[0][1]
                scale_y = image_rgb.shape[0] / images.image_sizes[0][0]

                x1, x2 = x1 * scale_x, x2 * scale_x
                y1, y2 = y1 * scale_y, y2 * scale_y

                # Draw rectangle
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=0.5, edgecolor='red', facecolor='none', alpha=0.6
                )
                ax.add_patch(rect)

            ax.set_title(f"Sample {current_sample} Anchors")
            ax.axis('off')

        # Hide unused axes
        for idx in range(len(anchor_densities) + 1, len(axes)):
            axes[idx].axis('off')

        plt.suptitle("Anchor Boxes Visualization", fontsize=16)
        plt.tight_layout()
        plt.savefig(anchor_dir / "anchor_boxes.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Anchor boxes saved to: {anchor_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize Faster R-CNN intermediate results')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Model checkpoint path')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--output', type=str, default='visualizations',
                       help='Output directory')
    parser.add_argument('--all', action='store_true',
                       help='Generate all visualizations')
    parser.add_argument('--feature-maps', action='store_true',
                       help='Generate feature maps')
    parser.add_argument('--rpn-attention', action='store_true',
                       help='Generate RPN attention maps')
    parser.add_argument('--heatmap', action='store_true',
                       help='Generate confidence heatmaps')
    parser.add_argument('--anchors', action='store_true',
                       help='Generate anchor boxes visualization')

    args = parser.parse_args()

    # If no specific visualization is selected, show all
    if not any([args.feature_maps, args.rpn_attention, args.heatmap, args.anchors]):
        args.all = True

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Get device
    device = get_device(config['device'])
    print(f"Using device: {device}")

    # Check model file
    if not os.path.exists(args.model):
        # Try to find the latest best model
        import glob
        model_files = glob.glob("checkpoints/*/best_model.pth")
        if model_files:
            args.model = max(model_files, key=os.path.getmtime)
            print(f"Using latest model: {args.model}")
        else:
            print(f"Error: Model file not found: {args.model}")
            print("Please train a model first or specify correct model path")
            return

    # Load model
    print(f"Loading model: {args.model}")
    model = load_pretrained_model(args.model, config, device)

    # Create visualizer
    visualizer = IntermediateVisualizer(model, config, device)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nVisualizing intermediate results for: {args.image}")
    print(f"Output directory: {output_dir}")

    # Generate visualizations
    if args.all or args.feature_maps:
        visualizer.visualize_feature_maps(args.image, output_dir)

    if args.all or args.rpn_attention:
        visualizer.visualize_rpn_attention(args.image, output_dir)

    if args.all or args.heatmap:
        visualizer.visualize_detection_heatmap(args.image, output_dir)

    if args.all or args.anchors:
        visualizer.visualize_anchor_boxes(args.image, output_dir)

    print("\nVisualization complete!")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()