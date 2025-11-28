#!/usr/bin/env python3
"""
YOLO11 Detection and Tracking Visualization Script

This script:
1. Runs YOLO11 detection on each image in the sequence
2. Compares predictions with ground truth labels
3. Sets up tracking for detected objects using ByteTrack
4. Visualizes the sequence with tracking over time
"""

import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
from ultralytics import YOLO


class LabelParser:
    """Parse KITTI format labels"""
    
    def __init__(self, labels_file):
        self.labels_file = labels_file
        self.labels_by_frame = defaultdict(list)
        self._parse_labels()
    
    def _parse_labels(self):
        """Parse labels file and organize by frame"""
        with open(self.labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                
                frame_id = int(parts[0])
                track_id = int(parts[1])
                class_name = parts[2]
                truncated = float(parts[3])
                occluded = int(parts[4])
                alpha = float(parts[5])
                
                # Bounding box (left, top, right, bottom)
                bbox = [float(parts[6]), float(parts[7]), 
                       float(parts[8]), float(parts[9])]
                
                # Dimensions (height, width, length)
                dimensions = [float(parts[10]), float(parts[11]), float(parts[12])]
                
                # Location (x, y, z)
                location = [float(parts[13]), float(parts[14]), float(parts[15])]
                
                # Rotation and score
                rotation_y = float(parts[16]) if len(parts) > 16 else 0.0
                score = float(parts[17]) if len(parts) > 17 else 1.0
                
                label = {
                    'frame_id': frame_id,
                    'track_id': track_id,
                    'class': class_name,
                    'truncated': truncated,
                    'occluded': occluded,
                    'alpha': alpha,
                    'bbox': bbox,  # [left, top, right, bottom]
                    'dimensions': dimensions,
                    'location': location,
                    'rotation_y': rotation_y,
                    'score': score
                }
                
                self.labels_by_frame[frame_id].append(label)
    
    def get_labels_for_frame(self, frame_id):
        """Get all labels for a specific frame"""
        return self.labels_by_frame.get(frame_id, [])


class YOLODetector:
    """YOLO11 detector wrapper with built-in tracking"""
    
    def __init__(self, model_name="yolo11s.pt", device=None):
        """
        Initialize YOLO11 model
        
        Args:
            model_name: Model name or path (e.g., 'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt')
            device: Device to run on ('cuda' or 'cpu'), None for auto
        """
        print(f"Loading YOLO11 model: {model_name}")
        self.model = YOLO(model_name)
        if device:
            self.model.to(device)
    
    def track(self, img, conf=0.25, iou=0.45, persist=True, tracker="bytetrack.yaml"):
        """
        Run detection and tracking on image
        
        Args:
            img: Input image (numpy array)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            persist: Whether to persist tracks across frames
            tracker: Tracker configuration file
        
        Returns:
            List of detections with tracking information
        """
        # Run YOLO tracking
        results = self.model.track(
            img, 
            conf=conf, 
            iou=iou, 
            persist=persist,
            tracker=tracker,
            classes=[0, 1, 2],  # Pedestrian/Cyclist/Car (or person/bicycle/car for COCO)
            verbose=False
        )
        
        # Extract detections from YOLO
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4]
                classes = result.boxes.cls.cpu().numpy()  # [N]
                confidences = result.boxes.conf.cpu().numpy()  # [N]
                track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None  # [N]
                
                for i in range(len(boxes)):
                    detection = {
                        'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                        'class': int(classes[i]),
                        'confidence': float(confidences[i]),
                        'track_id': int(track_ids[i]) if track_ids is not None else None
                    }
                    detections.append(detection)
        
        return detections


class KalmanBoxTracker:
    """Kalman filter for 2D bounding boxes with constant-acceleration model."""
    
    def __init__(self, bbox, process_noise=1e-2, measurement_noise=1e-1, dt=1.0):
        self.dt = float(dt)
        self._init_matrices(process_noise, measurement_noise)
        self.state = np.zeros((12, 1), dtype=np.float64)
        self.P = np.eye(12, dtype=np.float64) * 10.0
        self._set_state_from_bbox(bbox)
        self.last_prediction = bbox
    
    def _build_transition(self, dt):
        F = np.eye(12, dtype=np.float64)
        half_dt2 = 0.5 * dt * dt
        for i in range(4):
            pos = i
            vel = i + 4
            acc = i + 8
            F[pos, vel] = dt
            F[pos, acc] = half_dt2
            F[vel, acc] = dt
        return F
    
    def _init_matrices(self, process_noise, measurement_noise):
        self.F = self._build_transition(self.dt)
        self.H = np.zeros((4, 12), dtype=np.float64)
        self.H[:4, :4] = np.eye(4, dtype=np.float64)
        self.Q = np.eye(12, dtype=np.float64) * process_noise
        self.R = np.eye(4, dtype=np.float64) * measurement_noise
    
    def _bbox_to_measurement(self, bbox):
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        return np.array([[cx], [cy], [w], [h]], dtype=np.float64)
    
    def _measurement_to_bbox(self, measurement):
        cx, cy, w, h = measurement.flatten()
        w = max(1.0, w)
        h = max(1.0, h)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return [float(x1), float(y1), float(x2), float(y2)]
    
    def _set_state_from_bbox(self, bbox):
        measurement = self._bbox_to_measurement(bbox)
        self.state[:4] = measurement
        self.state[4:] = 0.0
        self.last_prediction = bbox
    
    def predict(self, dt=None):
        if dt is not None and dt != self.dt:
            self.dt = float(dt)
            self.F = self._build_transition(self.dt)
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.last_prediction = self._measurement_to_bbox(self.state[:4])
        return self.last_prediction
    
    def update(self, bbox):
        measurement = self._bbox_to_measurement(bbox)
        y = measurement - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = self.P @ self.H.T @ S_inv
        self.state = self.state + K @ y
        identity = np.eye(self.P.shape[0], dtype=np.float64)
        self.P = (identity - K @ self.H) @ self.P
        self.last_prediction = self._measurement_to_bbox(self.state[:4])
        return self.last_prediction


class OcclusionTracker:
    """
    Maintain track state during short occlusions using Kalman filters.
    
    Track IDs come from YOLO ByteTrack when available; otherwise synthetic IDs
    are assigned to keep predictions stable.
    """
    
    def __init__(self, max_occlusion_frames=8):
        self.max_occlusion_frames = max(1, max_occlusion_frames)
        self.trackers = {}
        self._next_internal_id = 10_000_000
    
    def _clip_bbox(self, bbox, frame_shape):
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0.0, min(x1, width - 1))
        y1 = max(0.0, min(y1, height - 1))
        x2 = max(0.0, min(x2, width - 1))
        y2 = max(0.0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]
    
    def _ensure_track_id(self, det):
        if det.get('track_id') is None:
            det['track_id'] = self._next_internal_id
            self._next_internal_id += 1
        return det['track_id']
    
    def step(self, detections, frame_shape):
        """
        Update trackers with current detections and return predicted boxes for
        tracks that went missing (occluded) for a limited number of frames.
        """
        if frame_shape is None:
            return []

        # Predict state for all existing trackers
        predictions = {}
        for track_id, tracker in self.trackers.items():
            predictions[track_id] = tracker.predict()
            tracker.missed_frames = getattr(tracker, 'missed_frames', 0) + 1

        matched_ids = set()

        for det in detections:
            track_id = self._ensure_track_id(det)
            tracker = self.trackers.get(track_id)
            if tracker is None:
                tracker = KalmanBoxTracker(det['bbox'])
                self.trackers[track_id] = tracker
                tracker.missed_frames = 0
            tracker.update(det['bbox'])
            tracker.missed_frames = 0
            tracker.class_id = det['class']
            tracker.last_confidence = det['confidence']
            predictions[track_id] = tracker.last_prediction
            matched_ids.add(track_id)

        occluded_predictions = []
        to_delete = []
        for track_id, tracker in self.trackers.items():
            if track_id in matched_ids:
                continue
            if tracker.missed_frames > self.max_occlusion_frames:
                to_delete.append(track_id)
                continue
            clipped_bbox = self._clip_bbox(predictions.get(track_id, tracker.last_prediction), frame_shape)
            if clipped_bbox is None:
                continue
            occluded_predictions.append({
                'bbox': clipped_bbox,
                'class': getattr(tracker, 'class_id', 0),
                'confidence': getattr(tracker, 'last_confidence', 0.0),
                'track_id': track_id,
                'occluded': True,
                'occlusion_age': tracker.missed_frames
            })

        for track_id in to_delete:
            self.trackers.pop(track_id, None)

        return occluded_predictions


class TrackKalmanFilter:
    """Kalman filter for smoothing 3D positions (X, Z) over time."""
    
    def __init__(self, initial_state, process_noise=1e-2, measurement_noise=1e-1):
        # state: [x, z, vx, vz]^T
        self.state = np.zeros((4, 1), dtype=np.float64)
        self.state[0, 0] = initial_state[0]
        self.state[1, 0] = initial_state[1]
        self.P = np.eye(4, dtype=np.float64)
        self.F = np.eye(4, dtype=np.float64)
        self.F[0, 2] = 1.0
        self.F[1, 3] = 1.0
        self.H = np.zeros((2, 4), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.Q = np.eye(4, dtype=np.float64) * process_noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise
        self.last_update_frame = None
    
    def predict(self, dt=1.0):
        F_dt = self.F.copy()
        F_dt[0, 2] = dt
        F_dt[1, 3] = dt
        self.state = F_dt @ self.state
        self.P = F_dt @ self.P @ F_dt.T + self.Q
        return self.state.copy()
    
    def update(self, measurement, frame_id=None):
        measurement = np.asarray(measurement, dtype=np.float64).reshape(2, 1)
        y = measurement - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = self.P @ self.H.T @ S_inv
        self.state = self.state + K @ y
        I = np.eye(self.P.shape[0], dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        self.last_update_frame = frame_id
        return self.state.copy()


class GlobalTrackSmoother:
    """Maintain Kalman filters per track ID to smooth stereo 3D positions."""
    
    def __init__(self, max_idle_frames=30, process_noise=1e-2, measurement_noise=1e-1):
        self.track_filters = {}
        self.max_idle_frames = max_idle_frames
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self._anon_id = 0
    
    def _get_tracker(self, track_id, detection):
        entry = self.track_filters.get(track_id)
        if entry is None:
            entry = {
                'kf': TrackKalmanFilter(
                    [detection['x'], detection['z']],
                    process_noise=self.process_noise,
                    measurement_noise=self.measurement_noise
                ),
                'last_seen': None,
                'last_update': None,
                'data': detection.copy(),
                'updated_frame': None
            }
            self.track_filters[track_id] = entry
        return entry
    
    def step(self, detections, frame_id):
        """
        detections: list of dicts with keys x,z,track_id,class,label
        Returns smoothed detections list with same dicts augmented.
        """
        smoothed = []
        for entry in self.track_filters.values():
            entry['updated_frame'] = None
        updated_ids = set()
        for det in detections:
            track_id = det.get('track_id')
            if track_id is None:
                track_id = f"anon_{self._anon_id}"
                self._anon_id += 1
            entry = self._get_tracker(track_id, det)
            dt = 1.0
            if entry['last_update'] is not None and frame_id is not None:
                dt = max(1.0, frame_id - entry['last_update'])
            entry['kf'].predict(dt=dt)
            entry['kf'].update([det['x'], det['z']], frame_id=frame_id)
            entry['last_seen'] = frame_id
            entry['last_update'] = frame_id
            entry['data'] = det.copy()
            entry['updated_frame'] = frame_id
            smoothed_state = entry['kf'].state.flatten()
            new_det = det.copy()
            new_det['track_id'] = track_id
            new_det['x'] = float(smoothed_state[0])
            new_det['z'] = float(smoothed_state[1])
            smoothed.append(new_det)
            updated_ids.add(track_id)

        self._cleanup_and_predict(frame_id, updated_ids)
        return smoothed
    
    def _cleanup_and_predict(self, frame_id, updated_ids):
        if frame_id is None:
            return
        to_delete = []
        for track_id, entry in self.track_filters.items():
            last_seen = entry['last_seen']
            if last_seen is None:
                continue
            if track_id not in updated_ids:
                dt = max(1.0, frame_id - (entry['last_update'] or frame_id))
                entry['kf'].predict(dt=dt)
                entry['last_update'] = frame_id
            if frame_id - last_seen > self.max_idle_frames:
                to_delete.append(track_id)
        for tid in to_delete:
            self.track_filters.pop(tid, None)
    
    def predict_only(self, frame_id):
        predictions = []
        for track_id, entry in self.track_filters.items():
            last_seen = entry['last_seen']
            if last_seen is None or frame_id - last_seen > self.max_idle_frames:
                continue
            state = entry['kf'].state.flatten()
            det = entry['data'].copy()
            det['track_id'] = track_id
            det['x'] = float(state[0])
            det['z'] = float(state[1])
            if entry.get('updated_frame') != frame_id:
                det['label'] = f"{det.get('label', 'Obj')} est"
            predictions.append(det)
        return predictions
    
    def touch_track(self, track_id, frame_id):
        entry = self.track_filters.get(track_id)
        if entry:
            entry['last_seen'] = frame_id


class DetectionSmoother:
    """Smooth per-camera 2D detections (cx, cy, w, h) using Kalman filters."""
    
    def __init__(self, max_idle_frames=20, process_noise=5e-3, measurement_noise=1.5e-1):
        self.filters = {}
        self.max_idle_frames = max_idle_frames
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
    
    def step(self, detections, frame_id=None):
        if not detections:
            return detections

        filtered = []
        for det in detections:
            track_id = det.get('track_id')
            if track_id is None:
                filtered.append(det)
                continue
            kf = self.filters.get(track_id)
            if kf is None:
                kf = KalmanBoxTracker(det['bbox'], process_noise=self.process_noise, measurement_noise=self.measurement_noise)
                self.filters[track_id] = kf
                kf.missed_frames = 0
            else:
                kf.predict()
                kf.update(det['bbox'])
                kf.missed_frames = 0
            smooth_bbox = kf.last_prediction
            new_det = det.copy()
            new_det['bbox'] = smooth_bbox
            filtered.append(new_det)

        to_delete = []
        for tid, kf in self.filters.items():
            kf.missed_frames = getattr(kf, 'missed_frames', 0) + 1
            if kf.missed_frames > self.max_idle_frames:
                to_delete.append(tid)
        for tid in to_delete:
            self.filters.pop(tid, None)

        return filtered


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes"""
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compare_detections_with_ground_truth(detections, ground_truth_labels):
    """Compare YOLO11 predictions with ground truth labels"""
    matches = []
    false_positives = []
    false_negatives = []
    
    # Normalize detections to a consistent dict structure
    pred_boxes = []
    for det in detections:
        pred_boxes.append({
            'bbox': det['bbox'],
            'class': det['class'],
            'confidence': det['confidence'],
            'track_id': det.get('track_id'),
            'occluded': det.get('occluded', False)
        })
    
    # Match predictions with ground truth
    matched_gt = set()
    matched_pred = set()
    
    for i, pred in enumerate(pred_boxes):
        pred_box = pred['bbox']
        pred_cls = pred['class']
        pred_conf = pred['confidence']
        track_id = pred['track_id']
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt_label in enumerate(ground_truth_labels):
            if j in matched_gt:
                continue
            
            gt_box = gt_label['bbox']  # [left, top, right, bottom]
            gt_class = gt_label['class']
            
            # Initialize class_match to False
            class_match = False
            
            # Map YOLO classes to KITTI classes
            # For custom trained model: 0=Pedestrian, 1=Cyclist, 2=Car
            # For COCO pretrained: 0=person, 1=bicycle, 2=car
            # Try custom model mapping first, then fallback to COCO
            if int(pred_cls) == 0 and gt_class == 'Pedestrian':
                class_match = True
            elif int(pred_cls) == 1 and gt_class == 'Cyclist':
                class_match = True
            elif int(pred_cls) == 2 and gt_class == 'Car':
                class_match = True
            else:
                # Fallback to COCO mapping
                class_map = {0: 'person', 1: 'bicycle', 2: 'car'}
                pred_class_name = class_map.get(int(pred_cls), 'unknown')
                if pred_class_name == 'person' and gt_class == 'Pedestrian':
                    class_match = True
                elif pred_class_name == 'bicycle' and gt_class == 'Cyclist':
                    class_match = True
                elif pred_class_name == 'car' and gt_class == 'Car':
                    class_match = True
            
            if class_match:
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        if best_iou > 0.3:  # IoU threshold
            match_entry = {
                'pred_idx': i,
                'gt_idx': best_gt_idx,
                'iou': best_iou,
                'pred': pred_boxes[i],
                'gt': ground_truth_labels[best_gt_idx]
            }
            matches.append(match_entry)
            matched_gt.add(best_gt_idx)
            matched_pred.add(i)
    
    # Find false positives and false negatives
    for i in range(len(pred_boxes)):
        if i not in matched_pred:
            false_positives.append(pred_boxes[i])
    
    for j in range(len(ground_truth_labels)):
        if j not in matched_gt:
            false_negatives.append(ground_truth_labels[j])
    
    return matches, false_positives, false_negatives


def draw_bbox(img, bbox, label, color, track_id=None, thickness=2):
    """Draw bounding box on image"""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Add label
    if track_id is not None:
        text = f"{label} ID:{track_id}"
    else:
        text = label
    
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.rectangle(img, (x1, y1 - text_height - baseline - 5), 
                  (x1 + text_width, y1), color, -1)
    cv2.putText(img, text, (x1, y1 - baseline - 2), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def visualize_frame(image, detections, ground_truth_labels, matches, 
                   false_positives, false_negatives, frame_id,
                   occluded_predictions=None, camera_name=None):
    """Visualize a single frame with detections and ground truth"""
    img = image.copy()
    
    # Draw ground truth (green)
    for gt in ground_truth_labels:
        bbox = gt['bbox']
        class_name = gt['class']
        track_id = gt['track_id']
        draw_bbox(img, bbox, f"GT:{class_name}", (0, 255, 0), track_id, 1)
    
    # Draw matched predictions (blue)
    for match in matches:
        pred_info = match['pred']
        pred_box = pred_info['bbox']
        pred_cls = pred_info['class']
        pred_conf = pred_info['confidence']
        track_id = pred_info.get('track_id')
        is_occluded = pred_info.get('occluded', False)
        # Map class ID to name (supports both custom and COCO models)
        class_map_custom = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car'}
        class_map_coco = {0: 'person', 1: 'bicycle', 2: 'car'}
        class_name = class_map_custom.get(int(pred_cls)) or class_map_coco.get(int(pred_cls), 'unknown')
        label_prefix = "OCC" if is_occluded else "P"
        color = (255, 0, 255) if is_occluded else (255, 0, 0)
        draw_bbox(img, pred_box, f"{label_prefix}:{class_name} {pred_conf:.2f}", 
                  color, track_id, 2)
    
    # Draw false positives (red)
    for fp in false_positives:
        pred_box = fp['bbox']
        pred_cls = fp['class']
        pred_conf = fp['confidence']
        track_id = fp.get('track_id')
        is_occluded = fp.get('occluded', False)
        class_map_custom = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car'}
        class_map_coco = {0: 'person', 1: 'bicycle', 2: 'car'}
        class_name = class_map_custom.get(int(pred_cls)) or class_map_coco.get(int(pred_cls), 'unknown')
        label_prefix = "OCC" if is_occluded else "FP"
        color = (200, 0, 255) if is_occluded else (0, 0, 255)
        draw_bbox(img, pred_box, f"{label_prefix}:{class_name}", color, track_id, 2)
    
    # Draw false negatives (yellow)
    for fn in false_negatives:
        bbox = fn['bbox']
        class_name = fn['class']
        draw_bbox(img, bbox, f"FN:{class_name}", (0, 255, 255), None, 1)
    
    # Draw predictions coming purely from the occlusion tracker (not scored)
    if occluded_predictions:
        class_map_custom = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car'}
        class_map_coco = {0: 'person', 1: 'bicycle', 2: 'car'}
        for occ in occluded_predictions:
            bbox = occ['bbox']
            pred_cls = occ.get('class', 0)
            track_id = occ.get('track_id')
            age = occ.get('occlusion_age', 0)
            class_name = class_map_custom.get(int(pred_cls)) or class_map_coco.get(int(pred_cls), 'unknown')
            draw_bbox(img, bbox, f"OCC:{class_name} Δ{age}", (255, 0, 255), track_id, 1)
    
    # Add frame info
    cv2.putText(img, f"Frame: {frame_id}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if camera_name:
        cv2.putText(img, f"Camera: {camera_name}", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        stats_y = 100
    else:
        stats_y = 70
    cv2.putText(img, f"TP: {len(matches)} | FP: {len(false_positives)} | FN: {len(false_negatives)}", 
               (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img


def bbox_center(bbox):
    """Return (cx, cy) for a bounding box"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def match_stereo_detections(left_detections, right_detections, baseline_m=0.54,
                            focal_length_px=721.0, max_vertical_diff=30.0,
                            min_disparity_px=1.0):
    """
    Match detections between left/right cameras to estimate distance using disparity.
    Returns (matches, unmatched_left, unmatched_right) where matches contain dicts with
    left/right detections, disparity, and distance (meters).
    """
    matches = []
    used_right = set()
    matched_left = set()
    
    for left_idx, left_det in enumerate(left_detections):
        best_idx = -1
        best_score = float('inf')
        best_match = None
        cx_left, cy_left = bbox_center(left_det['bbox'])

        for idx, right_det in enumerate(right_detections):
            if idx in used_right:
                continue
            if left_det['class'] != right_det['class']:
                continue
            
            cx_right, cy_right = bbox_center(right_det['bbox'])
            vertical_diff = abs(cy_left - cy_right)
            if vertical_diff > max_vertical_diff:
                continue
            
            disparity = cx_left - cx_right
            if disparity < min_disparity_px:
                continue
            
            score = vertical_diff + abs(disparity) * 0.01
            if score < best_score:
                best_score = score
                best_idx = idx
                best_match = {
                    'left': left_det,
                    'right': right_det,
                    'disparity': disparity
                }

        if best_idx >= 0 and best_match is not None:
            distance = (focal_length_px * baseline_m) / best_match['disparity']
            best_match['distance_m'] = float(distance)
            best_match['left_index'] = left_idx
            best_match['right_index'] = best_idx
            matches.append(best_match)
            used_right.add(best_idx)
            matched_left.add(left_idx)
    
    unmatched_left = [det for idx, det in enumerate(left_detections) if idx not in matched_left]
    unmatched_right = [det for idx, det in enumerate(right_detections) if idx not in used_right]
    
    return matches, unmatched_left, unmatched_right


def visualize_stereo_fusion(left_image, right_image, stereo_matches, frame_id,
                            camera_left_name, camera_right_name, stereo_params=None,
                            unmatched_left=None, unmatched_right=None):
    """
    Create a blended stereo view highlighting matched detections and estimated distances.
    If stereo_params is provided, annotate the panel with calibrated baseline/focal info.
    """
    if left_image is None or right_image is None:
        return None
    
    if left_image.shape != right_image.shape:
        right_image_resized = cv2.resize(right_image, (left_image.shape[1], left_image.shape[0]))
    else:
        right_image_resized = right_image
    
    fused = cv2.addWeighted(left_image, 0.5, right_image_resized, 0.5, 0)
    
    class_map_custom = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car'}
    class_map_coco = {0: 'person', 1: 'bicycle', 2: 'car'}
    
    for match in stereo_matches:
        left_bbox = match['left']['bbox']
        left_cls = match['left']['class']
        distance_m = match.get('distance_m', None)
        track_id = match['left'].get('track_id')
        class_name = class_map_custom.get(int(left_cls)) or class_map_coco.get(int(left_cls), 'unknown')
        label = f"{class_name}"
        if distance_m is not None:
            label += f" {distance_m:.1f}m"
        draw_bbox(fused, left_bbox, label, (0, 128, 255), track_id, 2)

        # Draw line between matched centers to illustrate disparity
        left_center = bbox_center(match['left']['bbox'])
        right_center = bbox_center(match['right']['bbox'])
        cv2.line(
            fused,
            (int(left_center[0]), int(left_center[1])),
            (int(right_center[0]), int(right_center[1])),
            (0, 255, 255),
            1
        )
    
    def draw_unmatched(detections, color, prefix):
        for det in detections or []:
            bbox = det.get('bbox')
            if bbox is None:
                continue
            det_cls = det.get('class', 0)
            track_id = det.get('track_id')
            class_name = class_map_custom.get(int(det_cls)) or class_map_coco.get(int(det_cls), 'unknown')
            label = f"{prefix}:{class_name}"
            if det.get('occluded'):
                label += " OCC"
            draw_bbox(fused, bbox, label, color, track_id, 1)
    
    draw_unmatched(unmatched_left, (0, 255, 0), 'L')
    draw_unmatched(unmatched_right, (0, 165, 255), 'R')
    
    cv2.putText(fused, f"Stereo Fusion: {camera_left_name} + {camera_right_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(fused, f"Matches: {len(stereo_matches)}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(fused, f"Frame: {frame_id}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if stereo_params and stereo_params.get('baseline_m'):
        cv2.putText(fused, f"Baseline: {stereo_params['baseline_m']:.3f} m", (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 255), 2)
    if stereo_params and stereo_params.get('focal_length_px'):
        cv2.putText(fused, f"Focal: {stereo_params['focal_length_px']:.1f} px", (10, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 255), 2)
    
    return fused


def assemble_multi_view(panels, layout='horizontal'):
    """
    Concatenate panels either horizontally or vertically.
    Panels that are None are skipped. Layout can be 'horizontal' or 'vertical'.
    """
    valid_panels = [p for p in panels if p is not None]
    if not valid_panels:
        return None
    
    heights = [p.shape[0] for p in valid_panels]
    widths = [p.shape[1] for p in valid_panels]
    channels = valid_panels[0].shape[2]
    dtype = valid_panels[0].dtype
    
    if layout == 'vertical':
        if len(set(widths)) > 1:
            target_width = min(widths)
            resized = []
            for panel in valid_panels:
                if panel.shape[1] != target_width:
                    scale = target_width / panel.shape[1]
                    new_height = int(panel.shape[0] * scale)
                    resized.append(cv2.resize(panel, (target_width, new_height)))
                else:
                    resized.append(panel)
            valid_panels = resized
            heights = [p.shape[0] for p in valid_panels]
            widths = [p.shape[1] for p in valid_panels]
        canvas_height = sum(heights)
        canvas_width = widths[0]
        canvas = np.zeros((canvas_height, canvas_width, channels), dtype=dtype)
        offset = 0
        for panel in valid_panels:
            panel_height = panel.shape[0]
            canvas[offset:offset+panel_height, :] = panel
            offset += panel_height
    else:
        if len(set(heights)) > 1:
            target_height = min(heights)
            resized = []
            for panel in valid_panels:
                if panel.shape[0] != target_height:
                    scale = target_height / panel.shape[0]
                    new_width = int(panel.shape[1] * scale)
                    resized.append(cv2.resize(panel, (new_width, target_height)))
                else:
                    resized.append(panel)
            valid_panels = resized
            heights = [p.shape[0] for p in valid_panels]
            widths = [p.shape[1] for p in valid_panels]
        canvas_width = sum(widths)
        canvas_height = heights[0]
        canvas = np.zeros((canvas_height, canvas_width, channels), dtype=dtype)
        offset = 0
        for panel in valid_panels:
            panel_width = panel.shape[1]
            canvas[:, offset:offset+panel_width] = panel
            offset += panel_width
    
    return canvas


def create_placeholder_panel(reference_image, message):
    """Create a placeholder panel with text when a view is unavailable."""
    if reference_image is None:
        return None
    panel = np.zeros_like(reference_image)
    y = 40
    for line in message.split('\n'):
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        y += 35
    return panel


def triangulate_stereo_match(match, stereo_params, fallback_principal=None):
    """
    Estimate 3D location (X, Y, Z) in left camera coordinates using disparity.
    fallback_principal is (cx, cy) in pixels when calibration principal point is unavailable.
    """
    if match is None or stereo_params is None:
        return None
    baseline = stereo_params.get('baseline_m', 0.0)
    focal = stereo_params.get('focal_length_px', 0.0)
    principal_point = stereo_params.get('principal_point') or fallback_principal
    disparity = match.get('disparity')
    if baseline <= 0 or focal <= 0 or principal_point is None:
        return None
    if disparity is None or disparity <= 0:
        return None
    
    cx_left, cy_left = bbox_center(match['left']['bbox'])
    px, py = principal_point
    Z = (focal * baseline) / disparity
    X = (cx_left - px) * Z / focal
    Y = (cy_left - py) * Z / focal
    return np.array([X, Y, Z], dtype=np.float32)


def create_topdown_map(detection_points, baseline_m, frame_id,
                       camera_left_name, camera_right_name,
                       lateral_range=20.0, depth_range=60.0,
                       width=500, height=600):
    """Render a top-down map showing camera baselines and 3D detections."""
    width = max(200, int(width))
    height = max(200, int(height))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    margin_x = 40
    margin_y = 40
    usable_width = width - 2 * margin_x
    usable_height = height - 2 * margin_y
    lateral_range = max(5.0, lateral_range)
    if baseline_m and baseline_m > lateral_range:
        lateral_range = baseline_m * 1.2
    depth_range = max(5.0, depth_range)
    
    def project_point(x, z):
        if z < 0:
            return None
        x_norm = (x + lateral_range) / (2 * lateral_range)
        x_norm = np.clip(x_norm, 0.0, 1.0)
        z_norm = min(z / depth_range, 1.0)
        px = int(margin_x + x_norm * usable_width)
        py = int(height - margin_y - z_norm * usable_height)
        return px, py
    
    # Draw grid lines
    for dz in np.arange(0, depth_range + 1e-6, 5.0):
        pt = project_point(0.0, dz)
        if pt is None:
            continue
        x0, y = margin_x, pt[1]
        x1 = width - margin_x
        color = (50, 50, 50) if dz % 10.0 else (80, 80, 80)
        cv2.line(canvas, (x0, y), (x1, y), color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{dz:.0f}m", (5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    
    for dx in np.arange(-lateral_range, lateral_range + 1e-6, 5.0):
        pt_top = project_point(dx, 0.0)
        pt_bottom = project_point(dx, depth_range)
        if pt_top is None or pt_bottom is None:
            continue
        color = (50, 50, 50) if abs(dx) % 10.0 else (80, 80, 80)
        cv2.line(canvas, (pt_top[0], pt_top[1]), (pt_bottom[0], pt_bottom[1]), color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{dx:.0f}", (pt_top[0]-10, height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    
    class_colors = {
        0: (0, 255, 0),
        1: (0, 200, 255),
        2: (0, 128, 255)
    }
    
    # Draw cameras
    left_cam_pt = project_point(0.0, 0.0)
    if left_cam_pt:
        cv2.circle(canvas, left_cam_pt, 6, (255, 255, 255), -1)
        cv2.putText(canvas, camera_left_name, (left_cam_pt[0]-30, left_cam_pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if baseline_m:
        right_cam_pt = project_point(baseline_m, 0.0)
        if right_cam_pt:
            cv2.circle(canvas, right_cam_pt, 6, (200, 200, 200), -1)
            cv2.putText(canvas, camera_right_name, (right_cam_pt[0]-30, right_cam_pt[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.line(canvas, left_cam_pt, right_cam_pt, (120, 120, 120), 2)
    
    # Draw detections
    for det in detection_points or []:
        x = det.get('x')
        z = det.get('z')
        cls = int(det.get('class', 0))
        pt = project_point(x, z)
        if pt is None:
            continue
        color = class_colors.get(cls, (255, 0, 0))
        cv2.circle(canvas, pt, 5, color, -1)
        label = det.get('label')
        if label:
            cv2.putText(canvas, label, (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    cv2.putText(canvas, f"Top-Down Map (Frame {frame_id})", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    legend_y = 50
    for cls_id, cls_name in [(0, 'Ped'), (1, 'Cyc'), (2, 'Car')]:
        color = class_colors.get(cls_id, (255, 0, 0))
        cv2.rectangle(canvas, (10, legend_y - 12), (30, legend_y + 4), color, -1)
        cv2.putText(canvas, cls_name, (35, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        legend_y += 20
    
    return canvas


def parse_calibration_file(calib_path):
    """Parse KITTI-style calibration file into a dict keyed by camera id."""
    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    
    raw_entries = {}
    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, values = line.split(':', 1)
            raw_entries[key.strip()] = values.strip()
    
    camera_data = {}
    
    def parse_numbers(value_str):
        return [float(x) for x in value_str.split()]
    
    for key, value in raw_entries.items():
        if '_' not in key:
            continue
        base, cam_id = key.rsplit('_', 1)
        if not cam_id.isdigit():
            continue
        camera_entry = camera_data.setdefault(cam_id, {})

        if base in {'P_rect', 'K', 'R', 'R_rect'}:
            rows = 3
            cols = 4 if base.startswith('P') else 3
            arr = np.array(parse_numbers(value), dtype=np.float64).reshape(rows, cols)
            camera_entry[base] = arr
        elif base in {'T'}:
            arr = np.array(parse_numbers(value), dtype=np.float64)
            camera_entry[base] = arr
        elif base in {'S', 'S_rect'}:
            arr = np.array(parse_numbers(value), dtype=np.float64)
            camera_entry[base] = arr
        elif base in {'D'}:
            camera_entry[base] = np.array(parse_numbers(value), dtype=np.float64)
    
    return camera_data


def get_camera_calibration(calibration_map, camera_name):
    """Return calibration entry (if any) for a given camera name (e.g., image_02)."""
    if calibration_map is None:
        return None
    if camera_name is None:
        return None
    if '_' in camera_name:
        cam_id = camera_name.split('_')[-1]
    else:
        cam_id = camera_name
    return calibration_map.get(cam_id)


def compute_stereo_parameters(left_calib, right_calib):
    """Derive stereo parameters (baseline, focal length, principal point)."""
    if left_calib is None or right_calib is None:
        return None
    
    P_left = left_calib.get('P_rect')
    if P_left is None:
        P_left = left_calib.get('K')
    P_right = right_calib.get('P_rect')
    if P_right is None:
        P_right = right_calib.get('K')
    T_left = left_calib.get('T')
    T_right = right_calib.get('T')
    
    if P_left is None or P_right is None or T_left is None or T_right is None:
        return None
    
    fx_left = P_left[0, 0]
    fx_right = P_right[0, 0]
    focal_length = float((fx_left + fx_right) / 2.0)
    
    baseline_vec = T_left[:3] - T_right[:3]
    baseline_m = float(np.linalg.norm(baseline_vec))
    
    principal_point = (
        float((P_left[0, 2] + P_right[0, 2]) / 2.0),
        float((P_left[1, 2] + P_right[1, 2]) / 2.0)
    )
    
    return {
        'baseline_m': baseline_m,
        'focal_length_px': focal_length,
        'principal_point': principal_point
    }


def maybe_resize_with_calibration(image, calibration_entry):
    """Resize image to match rectified size from calibration, if provided."""
    if calibration_entry is None:
        return image
    target_size = calibration_entry.get('S_rect')
    if target_size is None:
        target_size = calibration_entry.get('S')
    if target_size is None or len(target_size) < 2:
        return image
    target_width = int(target_size[0])
    target_height = int(target_size[1])
    if image.shape[1] == target_width and image.shape[0] == target_height:
        return image
    return cv2.resize(image, (target_width, target_height))


def extract_camera_intrinsics(calibration_entry):
    """Extract fx, fy, cx, cy and tx (baseline offset) from calibration entry."""
    if calibration_entry is None:
        return {}
    P = calibration_entry.get('P_rect')
    if P is None:
        P = calibration_entry.get('K')
    if P is None:
        return {}
    fx = float(P[0, 0])
    fy = float(P[1, 1])
    cx = float(P[0, 2])
    cy = float(P[1, 2])
    tx = float(P[0, 3] / fx) if fx != 0 and P.shape[1] > 3 else 0.0
    ty = float(P[1, 3] / fy) if fy != 0 and P.shape[1] > 3 else 0.0
    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'tx': tx, 'ty': ty}


def validate_calibration(camera_configs, stereo_params):
    if len(camera_configs) < 2:
        return
    left_cal = camera_configs[0].get('calibration')
    right_cal = camera_configs[1].get('calibration')
    if not left_cal or not right_cal:
        print("Warning: Missing calibration data for stereo validation")
        return
    T_left = left_cal.get('T')
    T_right = right_cal.get('T')
    if T_left is None or T_right is None:
        print("Warning: Translation vectors missing from calibration for stereo validation")
        return
    baseline_raw = float(np.linalg.norm(np.array(T_left[:3]) - np.array(T_right[:3])))
    diff = abs(baseline_raw - stereo_params['baseline_m'])
    if diff > 0.05:
        print(f"Warning: Baseline mismatch. Derived={stereo_params['baseline_m']:.3f}m vs calibration vectors {baseline_raw:.3f}m (diff {diff:.3f}m)")
    else:
        print(f"Calibration check passed. Baseline {stereo_params['baseline_m']:.3f}m (calib vectors {baseline_raw:.3f}m)")


def main():
    parser = argparse.ArgumentParser(description='YOLO11 Detection and Tracking Visualization')
    parser.add_argument('--seq_dir', type=str, 
                       default='34759_final_project_rect/seq_01',
                       help='Path to sequence directory')
    parser.add_argument('--camera', type=str, default='image_02',
                       choices=['image_02', 'image_03'],
                       help='Which camera to use (image_02 or image_03)')
    parser.add_argument('--save_video', action='store_true',
                       help='Save output as video file')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--show', action='store_true',
                       help='Show visualization in real-time')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--model', type=str, default='yolo11s.pt',
                       help='YOLO11 model name or path (e.g., yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.35,
                       help='IoU threshold for NMS')
    parser.add_argument('--ground_truth_only', action='store_true',
                       help='Only visualize ground truth labels (skip detection)')
    parser.add_argument('--dual_camera', action='store_true',
                       help='Process two synchronized cameras simultaneously')
    parser.add_argument('--camera_left', type=str, default='image_02',
                       help='Left camera name when using dual camera mode')
    parser.add_argument('--camera_right', type=str, default='image_03',
                       help='Right camera name when using dual camera mode')
    parser.add_argument('--baseline_meters', type=float, default=0.54,
                       help='Baseline distance between cameras in meters')
    parser.add_argument('--focal_length_px', type=float, default=721.0,
                       help='Approximate focal length in pixels for distance estimation')
    parser.add_argument('--max_stereo_vertical_diff', type=float, default=30.0,
                       help='Maximum vertical pixel difference when matching stereo detections')
    parser.add_argument('--min_stereo_disparity', type=float, default=1.0,
                       help='Minimum horizontal disparity (pixels) to accept a stereo match')
    parser.add_argument('--calib_file', type=str, default='34759_final_project_rect/calib_cam_to_cam.txt',
                       help='Path to KITTI-style calibration file for stereo parameter overrides')
    parser.add_argument('--disable_calibration_override', action='store_true',
                       help='Do not override baseline/focal length even if calibration is available')
    parser.add_argument('--map_depth_range', type=float, default=60.0,
                       help='Depth (meters) shown in the top-down map')
    parser.add_argument('--map_lateral_range', type=float, default=20.0,
                       help='Half-width (meters) shown left/right of the cameras in the map')
    parser.add_argument('--map_width', type=int, default=500,
                       help='Pixel width of the top-down map panel')
    parser.add_argument('--map_height', type=int, default=600,
                       help='Pixel height of the top-down map panel')
    parser.add_argument('--map_kalman_process_noise', type=float, default=1e-2,
                       help='Process noise for map-smoothing Kalman filter')
    parser.add_argument('--map_kalman_measurement_noise', type=float, default=1e-1,
                       help='Measurement noise for map-smoothing Kalman filter')
    parser.add_argument('--map_kalman_idle_frames', type=int, default=30,
                       help='Frames before dropping an idle map track')
    parser.add_argument('--det_kalman_process_noise', type=float, default=5e-3,
                       help='Process noise for per-detection Kalman smoother')
    parser.add_argument('--det_kalman_measurement_noise', type=float, default=1.5e-1,
                       help='Measurement noise for per-detection Kalman smoother')
    parser.add_argument('--det_kalman_idle_frames', type=int, default=10,
                       help='Idle frames before dropping a detection smoother track')
    parser.add_argument('--enable_occlusion_tracking', action='store_true',
                       help='Use Kalman filters to keep tracks alive through short occlusions')
    parser.add_argument('--max_occlusion_frames', type=int, default=8,
                       help='Maximum consecutive frames to predict during occlusion')
    parser.add_argument('--count_occluded_in_metrics', action='store_true',
                       help='Include occlusion tracker predictions when computing metrics')
    
    args = parser.parse_args()
    
    # Setup paths
    seq_dir = Path(args.seq_dir)
    labels_file = seq_dir / 'labels.txt'
    if args.dual_camera:
        camera_configs = [
            {
                'name': args.camera_left,
                'dir': seq_dir / args.camera_left / 'data'
            },
            {
                'name': args.camera_right,
                'dir': seq_dir / args.camera_right / 'data'
            }
        ]
    else:
        camera_configs = [{
            'name': args.camera,
            'dir': seq_dir / args.camera / 'data'
        }]
    primary_image_dir = camera_configs[0]['dir']
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract sequence name and camera for filename
    seq_name = seq_dir.name  # e.g., 'seq_01'
    if args.dual_camera:
        camera_tag = f"{camera_configs[0]['name']}_{camera_configs[1]['name']}"
    else:
        camera_tag = camera_configs[0]['name']
    
    calibration_data = None
    if args.calib_file:
        calib_path = Path(args.calib_file)
        if calib_path.exists():
            try:
                calibration_data = parse_calibration_file(calib_path)
                print(f"Loaded calibration data from {calib_path}")
            except Exception as exc:
                print(f"Warning: Failed to parse calibration file {calib_path}: {exc}")
        else:
            print(f"Warning: Calibration file not found: {calib_path}")
    
    camera_intrinsics = {}
    for cam_cfg in camera_configs:
        cam_cfg['calibration'] = get_camera_calibration(calibration_data, cam_cfg['name'])
        camera_intrinsics[cam_cfg['name']] = extract_camera_intrinsics(cam_cfg['calibration'])
    
    stereo_params = {
        'baseline_m': args.baseline_meters,
        'focal_length_px': args.focal_length_px,
        'principal_point': None,
        'calibration_used': False,
        'calibration_reference': None
    }
    if args.dual_camera and calibration_data:
        left_calib = camera_configs[0].get('calibration')
        right_calib = camera_configs[1].get('calibration')
        derived_params = compute_stereo_parameters(left_calib, right_calib)
        if derived_params:
            stereo_params['calibration_reference'] = derived_params
            if not args.disable_calibration_override:
                args.baseline_meters = derived_params['baseline_m']
                args.focal_length_px = derived_params['focal_length_px']
                print(f"Stereo parameters overridden by calibration: "
                      f"baseline={args.baseline_meters:.3f} m, focal={args.focal_length_px:.1f} px")
                stereo_params['calibration_used'] = True
                stereo_params['principal_point'] = derived_params.get('principal_point')
            else:
                stereo_params['principal_point'] = derived_params.get('principal_point')
    
    stereo_params['baseline_m'] = args.baseline_meters
    stereo_params['focal_length_px'] = args.focal_length_px
    if args.dual_camera:
        validate_calibration(camera_configs, stereo_params)
    
    map_config = {
        'depth_range': args.map_depth_range,
        'lateral_range': args.map_lateral_range,
        'width': args.map_width,
        'height': args.map_height
    }
    map_smoother = GlobalTrackSmoother(
        max_idle_frames=args.map_kalman_idle_frames,
        process_noise=args.map_kalman_process_noise,
        measurement_noise=args.map_kalman_measurement_noise
    )
    detection_smoothers = {}
    for cfg in camera_configs:
        detection_smoothers[cfg['name']] = DetectionSmoother(
            max_idle_frames=args.det_kalman_idle_frames,
            process_noise=args.det_kalman_process_noise,
            measurement_noise=args.det_kalman_measurement_noise
        )
    left_camera_name = camera_configs[0]['name']
    
    # Load YOLO11 model (only if not ground truth only mode)
    detectors = {}
    occlusion_trackers = {}
    if not args.ground_truth_only:
        print("Loading YOLO11 model...")
        for cfg in camera_configs:
            print(f"  -> Initializing detector for {cfg['name']}")
            detectors[cfg['name']] = YOLODetector(model_name=args.model)
        if args.enable_occlusion_tracking:
            for cfg in camera_configs:
                occlusion_trackers[cfg['name']] = OcclusionTracker(max_occlusion_frames=args.max_occlusion_frames)
            print(f"Occlusion tracking enabled (max gap = {args.max_occlusion_frames} frames)")
            if args.count_occluded_in_metrics:
                print("Occluded predictions will contribute to precision/recall metrics.")
        elif args.count_occluded_in_metrics:
            print("Warning: --count_occluded_in_metrics ignored because occlusion tracking is disabled.")
    else:
        print("Ground truth only mode: skipping detection")
    
    # Parse labels
    print("Parsing ground truth labels...")
    label_parser = LabelParser(labels_file)
    
    # Get all image files
    image_files = sorted(primary_image_dir.glob('*.png'))
    if args.max_frames:
        image_files = image_files[:args.max_frames]
    
    print(f"Found {len(image_files)} images")
    
    # Statistics
    stats_per_camera = defaultdict(lambda: {
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'iou_sum': 0.0,
        'matches': 0
    })
    stereo_stats = {'pairs': 0, 'distance_sum': 0.0}
    for cfg in camera_configs:
        _ = stats_per_camera[cfg['name']]
    
    # Process frames
    frames_visualized = []
    
    for idx, primary_img_path in enumerate(image_files):
        frame_id = int(primary_img_path.stem)
        print(f"Processing frame {frame_id} ({idx+1}/{len(image_files)})...")
        
        gt_labels = label_parser.get_labels_for_frame(frame_id)

        frame_images = {}
        camera_views = {}
        detections_for_stereo = {}
        skip_frame = False

        for cam_index, cam_cfg in enumerate(camera_configs):
            current_img_path = primary_img_path if cam_index == 0 else cam_cfg['dir'] / primary_img_path.name
            image = cv2.imread(str(current_img_path))
            if image is None:
                print(f"Warning: Could not load {current_img_path}")
                skip_frame = True
                break

            image = maybe_resize_with_calibration(image, cam_cfg.get('calibration'))
            frame_images[cam_cfg['name']] = image

            apply_ground_truth = not args.dual_camera or cam_index == 0
            cam_gt_labels = gt_labels if apply_ground_truth else []
            if args.ground_truth_only:
                detections = []
                matches = []
                false_positives = []
                false_negatives = []
                overlay_predictions = None
                stereo_input = []
            else:
                detector = detectors.get(cam_cfg['name'])
                if detector is None:
                    print(f"Warning: detector instance missing for camera {cam_cfg['name']}")
                    detections = []
                else:
                    detections = detector.track(image, conf=args.conf, iou=args.iou)
                smoother = detection_smoothers.get(cam_cfg['name'])
                if smoother:
                    detections = smoother.step(detections, frame_id=frame_id)
                tracker = occlusion_trackers.get(cam_cfg['name'])
                occluded_predictions = tracker.step(detections, image.shape) if tracker else []

                detections_for_eval = detections
                if tracker and args.count_occluded_in_metrics:
                    detections_for_eval = detections + occluded_predictions

                if cam_gt_labels:
                    matches, false_positives, false_negatives = compare_detections_with_ground_truth(
                        detections_for_eval, cam_gt_labels
                    )
                    stats_entry = stats_per_camera[cam_cfg['name']]
                    stats_entry['tp'] += len(matches)
                    stats_entry['fp'] += len(false_positives)
                    stats_entry['fn'] += len(false_negatives)
                    stats_entry['matches'] += len(matches)
                    stats_entry['iou_sum'] += sum(match['iou'] for match in matches)
                else:
                    display_detections = detections + occluded_predictions
                    matches = [
                        {
                            'pred_idx': idx,
                            'gt_idx': None,
                            'iou': None,
                            'pred': det
                        }
                        for idx, det in enumerate(display_detections)
                    ]
                    false_positives = []
                    false_negatives = []

                overlay_predictions = None if args.count_occluded_in_metrics else occluded_predictions
                if tracker and not args.count_occluded_in_metrics and occluded_predictions:
                    stereo_input = detections + occluded_predictions
                else:
                    stereo_input = detections_for_eval

                if cam_cfg['name'] == left_camera_name:
                    for det in detections + occluded_predictions:
                        track_id = det.get('track_id')
                        if track_id is not None:
                            map_smoother.touch_track(track_id, frame_id)

            if not args.ground_truth_only:
                detections_for_stereo[cam_cfg['name']] = stereo_input

            vis_image = visualize_frame(
                image,
                detections,
                cam_gt_labels,
                matches,
                false_positives,
                false_negatives,
                frame_id,
                occluded_predictions=overlay_predictions,
                camera_name=cam_cfg['name']
            )
            camera_views[cam_cfg['name']] = vis_image


        if skip_frame:
            continue

        combined_view = None
        if args.dual_camera:
            left_name = camera_configs[0]['name']
            right_name = camera_configs[1]['name']
            fusion_view = None
            stereo_matches = []
            unmatched_left = []
            unmatched_right = []
            map_measurements = []

            if not args.ground_truth_only:
                stereo_matches, unmatched_left, unmatched_right = match_stereo_detections(
                    detections_for_stereo.get(left_name, []),
                    detections_for_stereo.get(right_name, []),
                    baseline_m=stereo_params['baseline_m'],
                    focal_length_px=stereo_params['focal_length_px'],
                    max_vertical_diff=args.max_stereo_vertical_diff,
                    min_disparity_px=args.min_stereo_disparity
                )
            else:
                stereo_matches = []
                unmatched_left = []
                unmatched_right = []

            if stereo_matches:
                stereo_stats['pairs'] += len(stereo_matches)
                stereo_stats['distance_sum'] += sum(
                    m.get('distance_m', 0.0) for m in stereo_matches if m.get('distance_m') is not None
                )
                fallback_pp = None
                if stereo_params.get('principal_point') is None:
                    left_img = frame_images.get(left_name)
                    if left_img is not None:
                        h, w = left_img.shape[:2]
                        fallback_pp = (w / 2.0, h / 2.0)
                class_short_names = {0: 'Ped', 1: 'Cyc', 2: 'Car'}
                for match in stereo_matches:
                    position = triangulate_stereo_match(match, stereo_params, fallback_pp)
                    if position is None:
                        continue
                    match['position'] = position
                    label_distance = match.get('distance_m')
                    det_class = match['left']['class']
                    text_label = class_short_names.get(int(det_class), 'Obj')
                    if label_distance is not None:
                        text_label = f"{text_label} {label_distance:.1f}m"
                    track_id = match['left'].get('track_id')
                    if track_id is None:
                        continue
                    map_measurements.append({
                        'x': float(position[0]),
                        'z': float(position[2]),
                        'class': det_class,
                        'label': text_label,
                        'track_id': track_id
                    })

            fusion_view = visualize_stereo_fusion(
                frame_images.get(left_name),
                frame_images.get(right_name),
                stereo_matches,
                frame_id,
                left_name,
                right_name,
                stereo_params=stereo_params,
                unmatched_left=unmatched_left,
                unmatched_right=unmatched_right
            )
            if fusion_view is None:
                reference = frame_images.get(left_name)
                if reference is None:
                    reference = frame_images.get(right_name)
                fusion_view = create_placeholder_panel(reference, "Stereo fusion\nunavailable")
            map_smoother.step(map_measurements, frame_id)
            map_points = map_smoother.predict_only(frame_id)
            
            vertical_stack = assemble_multi_view([
                camera_views.get(left_name),
                camera_views.get(right_name),
                fusion_view
            ], layout='vertical')
            topdown_panel = create_topdown_map(
                map_points,
                stereo_params['baseline_m'],
                frame_id,
                left_name,
                right_name,
                lateral_range=map_config['lateral_range'],
                depth_range=map_config['depth_range'],
                width=map_config['width'],
                height=map_config['height']
            )
            combined_view = assemble_multi_view([
                vertical_stack,
                topdown_panel
            ], layout='horizontal')
        else:
            combined_view = camera_views.get(camera_configs[0]['name'])

        if combined_view is None:
            continue

        frames_visualized.append(combined_view)

        if args.show:
            cv2.imshow('Tracking Visualization', combined_view)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Print statistics
    print("\n" + "="*50)
    if args.ground_truth_only:
        print("GROUND TRUTH VISUALIZATION")
        print("="*50)
        print(f"Total Frames: {len(image_files)}")
        total_gt_objects = sum(len(label_parser.get_labels_for_frame(f)) for f in range(len(image_files)))
        print(f"Total Ground Truth Objects: {total_gt_objects}")
        precision = recall = f1 = 0.0  # Not applicable in GT-only mode
    else:
        total_tp = sum(stats['tp'] for stats in stats_per_camera.values())
        total_fp = sum(stats['fp'] for stats in stats_per_camera.values())
        total_fn = sum(stats['fn'] for stats in stats_per_camera.values())
        total_iou = sum(stats['iou_sum'] for stats in stats_per_camera.values())
        total_matches = sum(stats['matches'] for stats in stats_per_camera.values())
        avg_iou = total_iou / total_matches if total_matches > 0 else 0.0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print("DETECTION STATISTICS")
        print("="*50)
        print(f"Occlusion Tracking: {'ON' if args.enable_occlusion_tracking else 'OFF'}")
        if args.enable_occlusion_tracking:
            print(f"  Max Occlusion Frames: {args.max_occlusion_frames}")
            print(f"  Count Occluded In Metrics: {'Yes' if args.count_occluded_in_metrics else 'No'}")
        if args.dual_camera:
            print(f"Stereo Baseline: {args.baseline_meters:.3f} m | Focal Length: {args.focal_length_px:.1f} px")
            if stereo_params['calibration_used']:
                print("  (values derived from calibration file)")
            elif stereo_params['calibration_reference']:
                print("  (calibration file detected but override disabled)")
        print(f"Total Frames: {len(image_files)}")
        print(f"Overall True Positives: {total_tp}")
        print(f"Overall False Positives: {total_fp}")
        print(f"Overall False Negatives: {total_fn}")
        print(f"Overall Avg IoU: {avg_iou:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print("\nPer-Camera Metrics:")
        for cam_name, stats in stats_per_camera.items():
            cam_avg_iou = stats['iou_sum'] / stats['matches'] if stats['matches'] > 0 else 0.0
            print(f"  {cam_name}: TP={stats['tp']} FP={stats['fp']} FN={stats['fn']} AvgIoU={cam_avg_iou:.3f}")
        if args.dual_camera:
            if stereo_stats['pairs'] > 0:
                avg_distance = stereo_stats['distance_sum'] / stereo_stats['pairs']
                print(f"\nStereo Matches: {stereo_stats['pairs']} | Avg Distance: {avg_distance:.2f} m")
            else:
                print("\nStereo Matches: 0")
    print("="*50)
    
    # Save video if requested
    if args.save_video and frames_visualized:
        print("\nSaving video...")
        output_video = output_dir / f'tracking_visualization_{seq_name}_{camera_tag}.mp4'
        height, width = frames_visualized[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video), fourcc, 10.0, (width, height))
        
        for frame in frames_visualized:
            out.write(frame)
        out.release()
        print(f"Video saved to {output_video}")
    
    # Save statistics with sequence and camera name
    stats_file = output_dir / f'statistics_{seq_name}_{camera_tag}.txt'
    with open(stats_file, 'w') as f:
        if args.ground_truth_only:
            f.write("GROUND TRUTH VISUALIZATION\n")
            f.write("="*50 + "\n")
            f.write(f"Total Frames: {len(image_files)}\n")
            total_gt_objects = sum(len(label_parser.get_labels_for_frame(f)) for f in range(len(image_files)))
            f.write(f"Total Ground Truth Objects: {total_gt_objects}\n")
        else:
            f.write("DETECTION STATISTICS\n")
            f.write("="*50 + "\n")
            f.write(f"Occlusion Tracking: {'ON' if args.enable_occlusion_tracking else 'OFF'}\n")
            if args.enable_occlusion_tracking:
                f.write(f"Max Occlusion Frames: {args.max_occlusion_frames}\n")
                f.write(f"Occluded Counted In Metrics: {'Yes' if args.count_occluded_in_metrics else 'No'}\n")
            if args.dual_camera:
                f.write(f"Stereo Baseline: {args.baseline_meters:.3f} m\n")
                f.write(f"Focal Length: {args.focal_length_px:.1f} px\n")
                if stereo_params['calibration_used']:
                    f.write("Stereo Parameter Source: calibration file\n")
                elif stereo_params['calibration_reference']:
                    f.write("Stereo Parameter Source: user-provided defaults (calibration override disabled)\n")
                else:
                    f.write("Stereo Parameter Source: user-provided defaults\n")
            f.write(f"Total Frames: {len(image_files)}\n")
            f.write(f"True Positives: {total_tp}\n")
            f.write(f"False Positives: {total_fp}\n")
            f.write(f"False Negatives: {total_fn}\n")
            f.write(f"Average IoU: {avg_iou:.3f}\n")
            f.write(f"Precision: {precision:.3f}\n")
            f.write(f"Recall: {recall:.3f}\n")
            f.write(f"F1 Score: {f1:.3f}\n")
            f.write("\nPer-Camera Metrics:\n")
            for cam_name, stats in stats_per_camera.items():
                cam_avg_iou = stats['iou_sum'] / stats['matches'] if stats['matches'] > 0 else 0.0
                f.write(f"  {cam_name}: TP={stats['tp']} FP={stats['fp']} FN={stats['fn']} AvgIoU={cam_avg_iou:.3f}\n")
            if args.dual_camera:
                if stereo_stats['pairs'] > 0:
                    avg_distance = stereo_stats['distance_sum'] / stereo_stats['pairs']
                    f.write(f"\nStereo Matches: {stereo_stats['pairs']}\n")
                    f.write(f"Average Distance: {avg_distance:.2f} m\n")
                else:
                    f.write("\nStereo Matches: 0\n")
    
    print(f"\nStatistics saved to {stats_file}")
    
    if args.show:
        cv2.destroyAllWindows()
    
    print("\nDone!")


if __name__ == '__main__':
    main()

