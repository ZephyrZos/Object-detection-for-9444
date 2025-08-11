# Object Detection for Autonomous Drones

**Project ID: 041** - Computer Vision Implementation

This repository contains Python implementations of multiple object detection methods for autonomous drone applications using the VisDrone dataset.

## Methods Implemented

- **YOLOv5**: Single-stage detection with custom drone optimizations
- **UAV-DETR**: Transformer-based detection with UAV-specific modules  
- **Faster R-CNN**: Two-stage detection baseline
- **YOLOv8**: Latest YOLO architecture evaluation

## Repository Structure

```
Drone_ObjectDetection/
├── Yolo_method/          # YOLOv5 implementation 
├── UAV-DETR/             # UAV-DETR with custom ultralytics 
├── Faster_rcnn/          # Faster R-CNN implementation
├── Yolov8/               # YOLOv8 evaluation scripts
├── convert_*.py          # Dataset conversion utilities
└── analyze_*.py          # Dataset analysis tools
```

## Key Features

- **Multi-Method Comparison**: 5 different detection approaches
- **VisDrone Dataset Support**: Optimized for 10 drone object classes
- **Custom Training Scripts**: Method-specific optimization
- **Format Conversion**: VisDrone to YOLO/DETR format conversion
- **Analysis Tools**: Dataset statistics and visualization utilities

## Dataset
**VisDrone**: 6,471 training images, 548 validation images
**Classes**: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor

---

**Note**: This repository contains only Python source code (.py files). Model weights, datasets, and training outputs are excluded.