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
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment


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


def extract_appearance_features(img, bbox, model):
    """
    Extract appearance features from image region for ReID
    
    Args:
        img: Full image
        bbox: Bounding box [x1, y1, x2, y2]
        model: YOLO model for feature extraction
    
    Returns:
        Feature vector (normalized)
    """
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Extract ROI
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    
    # Resize ROI to fixed size for feature extraction
    roi_resized = cv2.resize(roi, (128, 256))  # Standard ReID size
    
    # Use YOLO's feature extractor if available, otherwise use simple histogram
    try:
        # Try to get features from YOLO model
        results = model.predict(roi_resized, verbose=False)
        if results and len(results) > 0:
            # Extract features from the model (if available)
            # For now, use a combination of color histogram and HOG-like features
            pass
    except:
        pass
    
    # Fallback: Extract color and texture features
    # Convert to different color spaces
    hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2LAB)
    
    # Color histograms
    hist_b = cv2.calcHist([roi_resized], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([roi_resized], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([roi_resized], [2], None, [32], [0, 256])
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    
    # Texture features (using L channel of LAB)
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    hist_gray = cv2.calcHist([gray], [0], None, [32], [0, 256])
    
    # Combine all features
    features = np.concatenate([
        hist_b.flatten(),
        hist_g.flatten(),
        hist_r.flatten(),
        hist_h.flatten(),
        hist_s.flatten(),
        hist_gray.flatten()
    ])
    
    # Normalize
    features = features / (np.linalg.norm(features) + 1e-8)
    
    return features


def compute_appearance_similarity(feat1, feat2):
    """
    Compute appearance similarity between two feature vectors
    
    Args:
        feat1: First feature vector
        feat2: Second feature vector
    
    Returns:
        Similarity score (0-1, higher is more similar)
    """
    if feat1 is None or feat2 is None:
        return 0.0
    
    # Use cosine similarity (1 - cosine distance)
    similarity = 1 - cosine(feat1, feat2)
    return max(0.0, similarity)


class KalmanTracker:
    """Kalman filter-based tracker for handling occlusions with ReID"""
    
    def __init__(self, track_id, bbox, class_id, appearance_feat=None, dt=1.0, std_acc=1.0, x_std_meas=0.1, y_std_meas=0.1):
        """
        Initialize Kalman filter for a track
        
        Args:
            track_id: Unique track identifier
            bbox: Initial bounding box [x1, y1, x2, y2]
            class_id: Object class ID
            appearance_feat: Appearance feature vector for ReID
            dt: Time step
            std_acc: Standard deviation of acceleration
            x_std_meas: Standard deviation of x measurement
            y_std_meas: Standard deviation of y measurement
        """
        self.track_id = track_id
        self.class_id = class_id
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        self.history = []
        self.appearance_features = []  # Store recent appearance features
        if appearance_feat is not None:
            self.appearance_features.append(appearance_feat)
        
        # Initialize Kalman filter with 8 state variables:
        # [x_center, y_center, width, height, vx, vy, vw, vh]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.eye(8)
        self.kf.F[0, 4] = dt  # x = x + vx*dt
        self.kf.F[1, 5] = dt  # y = y + vy*dt
        self.kf.F[2, 6] = dt  # w = w + vw*dt
        self.kf.F[3, 7] = dt  # h = h + vh*dt
        
        # Measurement matrix (we observe center and size)
        self.kf.H = np.zeros((4, 8))
        self.kf.H[0, 0] = 1  # observe x_center
        self.kf.H[1, 1] = 1  # observe y_center
        self.kf.H[2, 2] = 1  # observe width
        self.kf.H[3, 3] = 1  # observe height
        
        # Measurement noise
        self.kf.R = np.diag([x_std_meas**2, y_std_meas**2, x_std_meas**2, y_std_meas**2])
        
        # Process noise
        q = Q_discrete_white_noise(dim=2, dt=dt, var=std_acc**2)
        self.kf.Q = np.zeros((8, 8))
        self.kf.Q[0:2, 0:2] = q
        self.kf.Q[2:4, 2:4] = q
        self.kf.Q[4:6, 4:6] = q
        self.kf.Q[6:8, 6:8] = q
        
        # Initialize state: convert bbox to [cx, cy, w, h]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        
        self.kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.kf.P = np.eye(8) * 1000  # High initial uncertainty
        
        self.history.append(bbox)
    
    def update(self, bbox, appearance_feat=None):
        """Update Kalman filter with new detection"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        
        z = np.array([cx, cy, w, h], dtype=np.float32)
        self.kf.update(z)
        self.hits += 1
        self.time_since_update = 0
        
        # Update appearance features (keep last 5 for robustness)
        if appearance_feat is not None:
            self.appearance_features.append(appearance_feat)
            if len(self.appearance_features) > 5:
                self.appearance_features.pop(0)
        
        # Convert back to bbox format
        state = self.kf.x
        cx_pred, cy_pred, w_pred, h_pred = state[0], state[1], state[2], state[3]
        x1_pred = cx_pred - w_pred / 2
        y1_pred = cy_pred - h_pred / 2
        x2_pred = cx_pred + w_pred / 2
        y2_pred = cy_pred + h_pred / 2
        
        predicted_bbox = [x1_pred, y1_pred, x2_pred, y2_pred]
        self.history.append(predicted_bbox)
        return predicted_bbox
    
    def get_average_appearance(self):
        """Get average appearance feature from recent detections"""
        if len(self.appearance_features) == 0:
            return None
        return np.mean(self.appearance_features, axis=0)
    
    def predict(self):
        """Predict next state using Kalman filter"""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        
        # Convert state to bbox format
        state = self.kf.x
        cx_pred, cy_pred, w_pred, h_pred = state[0], state[1], state[2], state[3]
        x1_pred = cx_pred - w_pred / 2
        y1_pred = cy_pred - h_pred / 2
        x2_pred = cx_pred + w_pred / 2
        y2_pred = cy_pred + h_pred / 2
        
        predicted_bbox = [x1_pred, y1_pred, x2_pred, y2_pred]
        return predicted_bbox
    
    def get_state(self):
        """Get current state as bbox"""
        state = self.kf.x
        cx_pred, cy_pred, w_pred, h_pred = state[0], state[1], state[2], state[3]
        x1_pred = cx_pred - w_pred / 2
        y1_pred = cy_pred - h_pred / 2
        x2_pred = cx_pred + w_pred / 2
        y2_pred = cy_pred + h_pred / 2
        return [x1_pred, y1_pred, x2_pred, y2_pred]


class OcclusionTracker:
    """Enhanced tracker with Kalman filters and ReID for occlusion handling"""
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, 
                 appearance_threshold=0.7, reid_weight=0.5, model=None):
        """
        Initialize occlusion tracker
        
        Args:
            max_age: Maximum frames to keep a track without update
            min_hits: Minimum hits to confirm a track
            iou_threshold: IoU threshold for matching
            appearance_threshold: Minimum appearance similarity for ReID matching
            reid_weight: Weight for appearance similarity in matching (0-1)
            model: YOLO model for feature extraction
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.appearance_threshold = appearance_threshold
        self.reid_weight = reid_weight
        self.model = model
        self.trackers = {}  # track_id -> KalmanTracker
        self.frame_count = 0
        self.next_id = 1
    
    def update(self, detections, img=None):
        """
        Update tracker with new detections with ReID support
        
        Args:
            detections: List of detections, each with 'bbox', 'class', 'confidence', 'track_id'
            img: Full image for appearance feature extraction
        
        Returns:
            List of tracked objects with predicted positions for occluded ones
        """
        self.frame_count += 1
        
        # Extract appearance features for all detections
        det_features = {}
        if img is not None and self.model is not None:
            for i, det in enumerate(detections):
                feat = extract_appearance_features(img, det['bbox'], self.model)
                det_features[i] = feat
        
        # Convert detections to format: [bbox, class, conf, track_id, appearance_feat]
        dets = []
        for i, det in enumerate(detections):
            dets.append((
                det['bbox'], 
                det['class'], 
                det['confidence'], 
                det.get('track_id'),
                det_features.get(i)
            ))
        
        # Match detections with existing trackers using IoU + ReID
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, img)
        
        # Update matched trackers
        for m in matched:
            det_idx, trk_id = m
            bbox, cls, conf, _, appearance_feat = dets[det_idx]
            if trk_id in self.trackers:
                self.trackers[trk_id].update(bbox, appearance_feat)
        
        # Try to re-identify unmatched detections with occluded tracks using ReID
        reid_matched = self._reid_unmatched_detections(unmatched_dets, unmatched_trks, dets)
        
        # Update re-identified trackers
        for det_idx, trk_id in reid_matched:
            bbox, cls, conf, _, appearance_feat = dets[det_idx]
            if trk_id in self.trackers:
                self.trackers[trk_id].update(bbox, appearance_feat)
            # Remove from unmatched lists
            if det_idx in unmatched_dets:
                unmatched_dets.remove(det_idx)
            if trk_id in unmatched_trks:
                unmatched_trks.remove(trk_id)
        
        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            bbox, cls, conf, _, appearance_feat = dets[det_idx]
            trk_id = self.next_id
            self.next_id += 1
            self.trackers[trk_id] = KalmanTracker(trk_id, bbox, cls, appearance_feat)
        
        # Predict and handle occluded tracks
        tracked_objects = []
        tracks_to_remove = []
        
        for trk_id, tracker in self.trackers.items():
            if trk_id not in [m[1] for m in matched] and trk_id not in [m[1] for m in reid_matched]:
                # Track not matched - predict using Kalman filter
                predicted_bbox = tracker.predict()
                
                # Only keep if not too old and has enough hits
                if tracker.time_since_update < self.max_age and tracker.hits >= self.min_hits:
                    tracked_objects.append({
                        'bbox': predicted_bbox,
                        'class': tracker.class_id,
                        'confidence': 0.5,  # Lower confidence for predicted
                        'track_id': trk_id,
                        'occluded': True  # Mark as occluded
                    })
                elif tracker.time_since_update >= self.max_age:
                    tracks_to_remove.append(trk_id)
            else:
                # Track matched - use updated state
                bbox = tracker.get_state()
                tracked_objects.append({
                    'bbox': bbox,
                    'class': tracker.class_id,
                    'confidence': 0.9,  # High confidence for detected
                    'track_id': trk_id,
                    'occluded': False
                })
        
        # Remove old tracks
        for trk_id in tracks_to_remove:
            del self.trackers[trk_id]
        
        return tracked_objects
    
    def _associate_detections_to_trackers(self, detections, img=None):
        """Match detections to existing trackers using IoU + ReID"""
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(self.trackers.keys())
        
        trk_ids = list(self.trackers.keys())
        
        # Compute combined cost matrix (IoU + Appearance)
        cost_matrix = np.ones((len(detections), len(trk_ids))) * np.inf
        
        for d, det in enumerate(detections):
            det_bbox, det_cls, _, _, det_feat = det
            for t, trk_id in enumerate(trk_ids):
                tracker = self.trackers[trk_id]
                
                # Only match same class
                if det_cls != tracker.class_id:
                    continue
                
                predicted_bbox = tracker.get_state()
                iou = calculate_iou(det_bbox, predicted_bbox)
                
                # Compute appearance similarity if features available
                appearance_sim = 0.0
                if det_feat is not None:
                    trk_feat = tracker.get_average_appearance()
                    if trk_feat is not None:
                        appearance_sim = compute_appearance_similarity(det_feat, trk_feat)
                
                # Combined cost: lower is better
                # Use negative IoU and negative appearance similarity
                iou_cost = 1.0 - iou
                appearance_cost = 1.0 - appearance_sim
                combined_cost = (1 - self.reid_weight) * iou_cost + self.reid_weight * appearance_cost
                
                # Only consider if IoU is above threshold
                if iou > self.iou_threshold:
                    cost_matrix[d, t] = combined_cost
        
        # Use Hungarian algorithm for optimal assignment
        matched_indices = []
        if cost_matrix.size > 0 and not np.all(np.isinf(cost_matrix)):
            try:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_indices, col_indices):
                    if cost_matrix[r, c] < 1.0:  # Valid match
                        matched_indices.append((r, c))
            except:
                # Fallback to greedy matching
                pass
        
        # Convert to track IDs
        matched = [(d, trk_ids[t]) for d, t in matched_indices]
        matched_dets = [m[0] for m in matched]
        matched_trks = [m[1] for m in matched]
        
        unmatched_dets = [d for d in range(len(detections)) if d not in matched_dets]
        unmatched_trks = [trk_id for trk_id in trk_ids if trk_id not in matched_trks]
        
        return matched, unmatched_dets, unmatched_trks
    
    def _reid_unmatched_detections(self, unmatched_dets, unmatched_trks, detections):
        """
        Re-identify unmatched detections with occluded tracks using appearance features
        
        Args:
            unmatched_dets: List of unmatched detection indices
            unmatched_trks: List of unmatched tracker IDs
            detections: Full list of detections
        
        Returns:
            List of (detection_idx, tracker_id) pairs that were re-identified
        """
        reid_matches = []
        
        if len(unmatched_dets) == 0 or len(unmatched_trks) == 0:
            return reid_matches
        
        # Build cost matrix for ReID matching
        cost_matrix = np.ones((len(unmatched_dets), len(unmatched_trks))) * np.inf
        
        for i, det_idx in enumerate(unmatched_dets):
            det_bbox, det_cls, _, _, det_feat = detections[det_idx]
            
            if det_feat is None:
                continue
            
            for j, trk_id in enumerate(unmatched_trks):
                tracker = self.trackers[trk_id]
                
                # Only match same class
                if det_cls != tracker.class_id:
                    continue
                
                # Only try ReID for tracks that have been occluded for a while
                if tracker.time_since_update < 3:  # Don't ReID if just lost
                    continue
                
                trk_feat = tracker.get_average_appearance()
                if trk_feat is not None:
                    appearance_sim = compute_appearance_similarity(det_feat, trk_feat)
                    cost = 1.0 - appearance_sim
                    
                    # Only consider if appearance similarity is high enough
                    if appearance_sim > self.appearance_threshold:
                        cost_matrix[i, j] = cost
        
        # Use Hungarian algorithm for optimal ReID assignment
        if not np.all(np.isinf(cost_matrix)):
            try:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_indices, col_indices):
                    if cost_matrix[r, c] < (1.0 - self.appearance_threshold):
                        det_idx = unmatched_dets[r]
                        trk_id = unmatched_trks[c]
                        reid_matches.append((det_idx, trk_id))
            except:
                pass
        
        return reid_matches


class YOLODetector:
    """YOLO11 detector wrapper with built-in tracking and occlusion handling"""
    
    def __init__(self, model_name="yolo11l.pt", device=None, use_occlusion_tracking=True):
        """
        Initialize YOLO11 model
        
        Args:
            model_name: Model name or path (e.g., 'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt')
            device: Device to run on ('cuda' or 'cpu'), None for auto
            use_occlusion_tracking: Whether to use Kalman filter for occlusion handling
        """
        print(f"Loading YOLO11 model: {model_name}")
        self.model = YOLO(model_name)
        if device:
            self.model.to(device)
        
        self.use_occlusion_tracking = use_occlusion_tracking
        if use_occlusion_tracking:
            self.occlusion_tracker = OcclusionTracker(
                max_age=30, 
                min_hits=3, 
                iou_threshold=0.3,
                appearance_threshold=0.7,
                reid_weight=0.5,
                model=self.model
            )
            print("Occlusion tracking with Kalman filters and ReID enabled")
    
    def track(self, img, conf=0.25, iou=0.45, persist=True, tracker="bytetrack.yaml"):
        """
        Run detection and tracking on image with occlusion handling
        
        Args:
            img: Input image (numpy array)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            persist: Whether to persist tracks across frames
            tracker: Tracker configuration file
        
        Returns:
            List of detections with tracking information (including occluded predictions)
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
        
        # Apply occlusion tracking with Kalman filters and ReID
        if self.use_occlusion_tracking:
            tracked_objects = self.occlusion_tracker.update(detections, img)
            return tracked_objects
        
        return detections


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
    
    # Convert detections to list of (bbox, class, conf, track_id, occluded)
    pred_boxes = []
    for det in detections:
        bbox = det['bbox']  # [x1, y1, x2, y2]
        cls = det['class']
        conf = det['confidence']
        track_id = det.get('track_id', None)
        occluded = det.get('occluded', False)
        pred_boxes.append((bbox, cls, conf, track_id, occluded))
    
    # Match predictions with ground truth
    matched_gt = set()
    matched_pred = set()
    
    for i, (pred_box, pred_cls, pred_conf, track_id, occluded) in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt_label in enumerate(ground_truth_labels):
            if j in matched_gt:
                continue
            
            gt_box = gt_label['bbox']  # [left, top, right, bottom]
            gt_class = gt_label['class']
            
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
            matches.append({
                'pred_idx': i,
                'gt_idx': best_gt_idx,
                'iou': best_iou,
                'pred': (pred_box, pred_cls, pred_conf, track_id, occluded),
                'gt': ground_truth_labels[best_gt_idx]
            })
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


def draw_bbox(img, bbox, label, color, track_id=None, thickness=2, occluded=False):
    """Draw bounding box on image"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Use dashed line for occluded objects
    if occluded:
        # Draw dashed rectangle
        dash_length = 10
        gap_length = 5
        # Top
        x = x1
        while x < x2:
            end_x = min(x + dash_length, x2)
            cv2.line(img, (x, y1), (end_x, y1), color, thickness)
            x += dash_length + gap_length
        # Bottom
        x = x1
        while x < x2:
            end_x = min(x + dash_length, x2)
            cv2.line(img, (x, y2), (end_x, y2), color, thickness)
            x += dash_length + gap_length
        # Left
        y = y1
        while y < y2:
            end_y = min(y + dash_length, y2)
            cv2.line(img, (x1, y), (x1, end_y), color, thickness)
            y += dash_length + gap_length
        # Right
        y = y1
        while y < y2:
            end_y = min(y + dash_length, y2)
            cv2.line(img, (x2, y), (x2, end_y), color, thickness)
            y += dash_length + gap_length
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Add label
    if track_id is not None:
        text = f"{label} ID:{track_id}"
        if occluded:
            text += " [OCC]"
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
                   false_positives, false_negatives, frame_id):
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
        pred_box = match['pred'][0]
        pred_cls = match['pred'][1]
        pred_conf = match['pred'][2]
        track_id = match['pred'][3]
        is_occluded = match['pred'][4] if len(match['pred']) > 4 else False
        # Map class ID to name (supports both custom and COCO models)
        class_map_custom = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car'}
        class_map_coco = {0: 'person', 1: 'bicycle', 2: 'car'}
        class_name = class_map_custom.get(int(pred_cls)) or class_map_coco.get(int(pred_cls), 'unknown')
        draw_bbox(img, pred_box, f"P:{class_name} {pred_conf:.2f}", 
                  (255, 0, 0), track_id, 2, occluded=is_occluded)
    
    # Draw false positives (red)
    for fp in false_positives:
        pred_box = fp[0]
        pred_cls = fp[1]
        pred_conf = fp[2]
        track_id = fp[3] if len(fp) > 3 else None
        is_occluded = fp[4] if len(fp) > 4 else False
        class_map_custom = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car'}
        class_map_coco = {0: 'person', 1: 'bicycle', 2: 'car'}
        class_name = class_map_custom.get(int(pred_cls)) or class_map_coco.get(int(pred_cls), 'unknown')
        draw_bbox(img, pred_box, f"FP:{class_name}", (0, 0, 255), track_id, 2, occluded=is_occluded)
    
    # Draw false negatives (yellow)
    for fn in false_negatives:
        bbox = fn['bbox']
        class_name = fn['class']
        draw_bbox(img, bbox, f"FN:{class_name}", (0, 255, 255), None, 1)
    
    # Add frame info
    cv2.putText(img, f"Frame: {frame_id}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"TP: {len(matches)} | FP: {len(false_positives)} | FN: {len(false_negatives)}", 
               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img


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
    parser.add_argument('--model', type=str, default='yolo11l.pt',
                       help='YOLO11 model name or path (e.g., yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.35,
                       help='IoU threshold for NMS')
    parser.add_argument('--no_occlusion_tracking', action='store_true',
                       help='Disable Kalman filter-based occlusion tracking')
    parser.add_argument('--ground_truth_only', action='store_true',
                       help='Only visualize ground truth labels (skip detection)')
    
    args = parser.parse_args()
    
    # Setup paths
    seq_dir = Path(args.seq_dir)
    labels_file = seq_dir / 'labels.txt'
    image_dir = seq_dir / args.camera / 'data'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract sequence name and camera for filename
    seq_name = seq_dir.name  # e.g., 'seq_01'
    camera_name = args.camera  # e.g., 'image_02'
    
    # Load YOLO11 model (only if not ground truth only mode)
    detector = None
    if not args.ground_truth_only:
        print("Loading YOLO11 model...")
        detector = YOLODetector(model_name=args.model, use_occlusion_tracking=not args.no_occlusion_tracking)
    else:
        print("Ground truth only mode: skipping detection")
    
    # Parse labels
    print("Parsing ground truth labels...")
    label_parser = LabelParser(labels_file)
    
    # Get all image files
    image_files = sorted(image_dir.glob('*.png'))
    if args.max_frames:
        image_files = image_files[:args.max_frames]
    
    print(f"Found {len(image_files)} images")
    
    # Statistics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0
    num_matches = 0
    
    # Process frames
    frames_visualized = []
    
    for idx, img_path in enumerate(image_files):
        frame_id = int(img_path.stem)
        print(f"Processing frame {frame_id} ({idx+1}/{len(image_files)})...")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not load {img_path}")
            continue
        
        # Get ground truth labels for this frame
        gt_labels = label_parser.get_labels_for_frame(frame_id)
        
        if args.ground_truth_only:
            # Only visualize ground truth
            detections = []
            matches = []
            false_positives = []
            false_negatives = []
            vis_image = visualize_frame(image, detections, gt_labels, matches,
                                      false_positives, false_negatives, frame_id)
        else:
            # Run YOLO11 detection with tracking
            detections = detector.track(image, conf=args.conf, iou=args.iou)
            
            # Compare with ground truth
            matches, false_positives, false_negatives = compare_detections_with_ground_truth(
                detections, gt_labels
            )
            
            # Update statistics
            total_tp += len(matches)
            total_fp += len(false_positives)
            total_fn += len(false_negatives)
            for match in matches:
                total_iou += match['iou']
                num_matches += 1
            
            # Visualize frame
            vis_image = visualize_frame(image, detections, gt_labels, matches,
                                      false_positives, false_negatives, frame_id)
        frames_visualized.append(vis_image)
        
        # Show if requested
        if args.show:
            cv2.imshow('Tracking Visualization', vis_image)
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
        print("DETECTION STATISTICS")
        print("="*50)
        print(f"Total Frames: {len(image_files)}")
        print(f"True Positives: {total_tp}")
        print(f"False Positives: {total_fp}")
        print(f"False Negatives: {total_fn}")
        if num_matches > 0:
            avg_iou = total_iou / num_matches
            print(f"Average IoU: {avg_iou:.3f}")
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
    print("="*50)
    
    # Save video if requested
    if args.save_video and frames_visualized:
        print("\nSaving video...")
        output_video = output_dir / f'tracking_visualization_{seq_name}_{camera_name}.mp4'
        height, width = frames_visualized[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video), fourcc, 10.0, (width, height))
        
        for frame in frames_visualized:
            out.write(frame)
        out.release()
        print(f"Video saved to {output_video}")
    
    # Save statistics with sequence and camera name
    stats_file = output_dir / f'statistics_{seq_name}_{camera_name}.txt'
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
            f.write(f"Total Frames: {len(image_files)}\n")
            f.write(f"True Positives: {total_tp}\n")
            f.write(f"False Positives: {total_fp}\n")
            f.write(f"False Negatives: {total_fn}\n")
            if num_matches > 0:
                f.write(f"Average IoU: {total_iou / num_matches:.3f}\n")
            f.write(f"Precision: {precision:.3f}\n")
            f.write(f"Recall: {recall:.3f}\n")
            f.write(f"F1 Score: {f1:.3f}\n")
    
    print(f"\nStatistics saved to {stats_file}")
    
    if args.show:
        cv2.destroyAllWindows()
    
    print("\nDone!")


if __name__ == '__main__':
    main()

