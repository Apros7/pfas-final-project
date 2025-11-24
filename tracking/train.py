#!/usr/bin/env python3
"""
Train YOLO11s on KITTI dataset using Ultralytics
"""

from ultralytics import YOLO

# Initialize model
model = YOLO('yolo11s.pt')

# Train the model
results = model.train(
    data='kitti.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    project='runs/detect',
    name='yolo11s_kitti',
    patience=50,
    save=True,
    save_period=10,
    val=True,
    plots=True
)

print(f"\n✓ Training complete! Best model: {model.trainer.best}")

