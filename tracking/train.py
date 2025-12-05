#!/usr/bin/env python3
"""
Train YOLO11s on KITTI dataset using Ultralytics
"""

from ultralytics import YOLO

# Initialize model
model = YOLO('yolo11s.pt')

# Train the model with memory-efficient settings
results = model.train(
    data='kitti.yaml',
    epochs=100,
    imgsz=640,
    batch=8,  # Reduced from 16 to save memory
    workers=4,  # Reduce data loading workers to save memory
    amp=True,  # Enable mixed precision training (uses less memory)
    project='runs/detect',
    name='yolo11s_kitti',
    patience=50,
    save=True,
    save_period=10,
    val=True,
    plots=True
)

print(f"\n✓ Training complete! Best model: {model.trainer.best}")

