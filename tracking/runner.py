from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from camera import create_stereo_matcher
from constants import YOLO_MODEL_PATH
from depth_estimator import (
    compute_disparity_map,
    estimate_depth_from_single_camera,
    estimate_depth_from_stereo_match,
)
from detector import ObjectDetector
from kalman_tracker import KalmanTracker
from stereo_matcher import match_detections
from track_state import TrackState
from types import PaintReturn, Point3D
from visualizer import (
    draw_annotations,
    draw_fusion_view,
    draw_ground_truth,
    draw_refined_predictions,
)


class Runner:
    def __init__(
        self,
        *,
        model_path: str = YOLO_MODEL_PATH,
        conf: float = 0.25,
        iou: float = 0.5,
        device: str | None = None,
        seq_dir: str,
        fx: float | None,
        fy: float | None,
        cx: float | None,
        cy: float | None,
        baseline: float,
    ):
        self.detector = ObjectDetector(model_path, conf, iou, device)
        self.seq_dir = Path(seq_dir)
        self.labels_by_frame = self._load_labels(self.seq_dir / "labels.txt")
        self.frame_names = self._collect_frame_names(self.seq_dir)
        self._frame_idx = 0
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.baseline = baseline
        self.tracker = KalmanTracker(max_idle=15, min_hits=3, match_threshold=4.0)
        self._sgbm = create_stereo_matcher(fx, fy, cx, cy, baseline)
        self._disparity_cache: Dict[int, np.ndarray] = {}

    def __call__(self, img_left: np.ndarray, img_right: np.ndarray) -> PaintReturn:
        return self.paint(img_left, img_right)

    def paint(self, img_left: np.ndarray, img_right: np.ndarray) -> PaintReturn:
        frame_id = self._current_frame_id()
        left_overlay, left_dets = self._annotate(img_left, (0, 170, 255))
        right_overlay, right_dets = self._annotate(img_right, (255, 120, 0))
        gt_view = draw_ground_truth(
            img_left.copy(), self.labels_by_frame.get(frame_id, []), frame_id
        )
        self._advance_frame()

        disparity = self._compute_disparity(frame_id, img_left, img_right)
        left_points, right_points, predicted_points, refined = self._estimate_depth(
            left_dets, right_dets, disparity, frame_id
        )
        refined_view = draw_refined_predictions(img_left.copy(), refined, frame_id)
        fused_view = draw_fusion_view(img_left.copy(), refined, frame_id)

        images = (left_overlay, right_overlay, refined_view, fused_view, gt_view)
        cloud = (left_points, right_points, predicted_points)
        return images, cloud

    def _annotate(
        self, image: np.ndarray, color: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, List]:
        detections = self.detector.detect(image)
        overlay = draw_annotations(image, detections, color)
        return overlay, detections

    def _compute_disparity(
        self, frame_id: int | None, left_img: np.ndarray, right_img: np.ndarray
    ) -> Optional[np.ndarray]:
        if self._sgbm is None or frame_id is None:
            return None
        if frame_id in self._disparity_cache:
            return self._disparity_cache[frame_id]
        disparity = compute_disparity_map(left_img, right_img, self._sgbm)
        if disparity is not None:
            if len(self._disparity_cache) > 30:
                self._disparity_cache.pop(next(iter(self._disparity_cache)))
            self._disparity_cache[frame_id] = disparity
        return disparity

    def _estimate_depth(
        self,
        left_dets: List,
        right_dets: List,
        disparity_map: Optional[np.ndarray],
        frame_id: Optional[int],
    ) -> Tuple[List[Point3D], List[Point3D], List[Point3D], List[Dict[str, float]]]:
        self.tracker.predict()

        if (
            not left_dets
            or not right_dets
            or disparity_map is None
            or not self._has_calibration()
        ):
            active_tracks, _ = self.tracker.update([])
            left_points = [(state.x, state.y, state.z) for state in active_tracks]
            right_points = [
                (state.x - self.baseline, state.y, state.z) for state in active_tracks
            ]
            return left_points, right_points, [], []

        fx = self.fx or 0.0
        fy = self.fy or fx
        cx = self.cx or 0.0
        cy = self.cy or 0.0

        measurements: List[Dict[str, float]] = []
        refined_detections: List[Dict[str, float]] = []

        matches = match_detections(left_dets, right_dets)
        matched_left_indices = set()
        matched_right_indices = set()

        for det_left, det_right in matches:
            for i, det in enumerate(left_dets):
                if det is det_left:
                    matched_left_indices.add(i)
                    break
            for i, det in enumerate(right_dets):
                if det is det_right:
                    matched_right_indices.add(i)
                    break

            measurement = estimate_depth_from_stereo_match(
                det_left,
                det_right,
                disparity_map,
                fx,
                fy,
                cx,
                cy,
                self.baseline,
            )
            if measurement:
                measurements.append(
                    {"x": measurement["x"], "y": measurement["y"], "z": measurement["z"], "class": measurement["class"], "stereo": True}
                )
                measurement["frame_id"] = frame_id
                refined_detections.append(measurement)

        for i, det in enumerate(left_dets):
            if i not in matched_left_indices and disparity_map is not None:
                measurement = estimate_depth_from_single_camera(
                    det, disparity_map, fx, fy, cx, cy, self.baseline, False
                )
                if measurement:
                    measurements.append(
                        {"x": measurement["x"], "y": measurement["y"], "z": measurement["z"], "class": measurement["class"], "stereo": False}
                    )
                    measurement["frame_id"] = frame_id
                    refined_detections.append(measurement)

        for i, det in enumerate(right_dets):
            if i not in matched_right_indices and disparity_map is not None:
                measurement = estimate_depth_from_single_camera(
                    det, disparity_map, fx, fy, cx, cy, self.baseline, True
                )
                if measurement:
                    measurements.append(
                        {"x": measurement["x"], "y": measurement["y"], "z": measurement["z"], "class": measurement["class"], "stereo": False}
                    )
                    measurement["frame_id"] = frame_id
                    refined_detections.append(measurement)

        active_tracks, matched_measurements = self.tracker.update(measurements)

        track_id_to_state = {state.track_id: state for state in active_tracks}
        measurement_to_track: Dict[int, TrackState] = {}

        for i, measurement in enumerate(measurements):
            meas_pos = np.array([measurement["x"], measurement["y"], measurement["z"]])
            best_track = None
            best_dist = float("inf")

            for state in active_tracks:
                if (
                    state.last_seen == self.tracker._frame_idx
                    and state.frames_since_last_detection == 0
                ):
                    track_pos = np.array([state.x, state.y, state.z])
                    dist = np.linalg.norm(meas_pos - track_pos)
                    if dist < best_dist and dist < 1.0 and state.cls_id == measurement["class"]:
                        best_dist = dist
                        best_track = state

            if best_track is None:
                for state in active_tracks:
                    track_pos = np.array([state.x, state.y, state.z])
                    dist = np.linalg.norm(meas_pos - track_pos)
                    if dist < best_dist and dist < 2.0 and state.cls_id == measurement["class"]:
                        best_dist = dist
                        best_track = state

            if best_track is not None:
                measurement_to_track[i] = best_track

        filtered_refined_detections = []
        track_id_to_state = {state.track_id: state for state in active_tracks}

        for i, det in enumerate(refined_detections):
            if i < len(measurements) and i in measurement_to_track:
                track = measurement_to_track[i]
                det["track_id"] = track.track_id
                det["x"] = track.x
                det["y"] = track.y
                det["z"] = track.z
                det["depth"] = track.z

                if "bbox" in det:
                    track.last_bbox = np.array(det["bbox"], dtype=np.float32)

                is_single_camera = det.get("single_camera", False) or not det.get(
                    "stereo", True
                )
                if is_single_camera:
                    if track.stereo_detections > 5 and track.single_camera_frames <= 3:
                        det["established_track"] = True
                        filtered_refined_detections.append(det)
                else:
                    det["established_track"] = track.stereo_detections > 5
                    filtered_refined_detections.append(det)
            else:
                det_pos = np.array(
                    [det.get("x", 0), det.get("y", 0), det.get("z", 0)]
                )
                best_track = None
                best_dist = float("inf")

                for state in active_tracks:
                    track_pos = np.array([state.x, state.y, state.z])
                    dist = np.linalg.norm(det_pos - track_pos)
                    if dist < best_dist and dist < 3.0:
                        best_dist = dist
                        best_track = state

                if best_track is not None:
                    det["track_id"] = best_track.track_id
                    det["x"] = best_track.x
                    det["y"] = best_track.y
                    det["z"] = best_track.z
                    det["depth"] = best_track.z

                    if "bbox" in det:
                        best_track.last_bbox = np.array(det["bbox"], dtype=np.float32)

                    is_single_camera = det.get("single_camera", False) or not det.get(
                        "stereo", True
                    )
                    if is_single_camera:
                        if (
                            best_track.stereo_detections > 5
                            and best_track.single_camera_frames <= 3
                        ):
                            det["established_track"] = True
                            filtered_refined_detections.append(det)
                    else:
                        det["established_track"] = best_track.stereo_detections > 5
                        filtered_refined_detections.append(det)
                else:
                    filtered_refined_detections.append(det)

        fx = self.fx or 0.0
        fy = self.fy or fx
        cx = self.cx or 0.0
        cy = self.cy or 0.0

        matched_track_ids = {
            det.get("track_id")
            for det in filtered_refined_detections
            if det.get("track_id") is not None
        }

        for track_id, (kf, state) in self.tracker.tracks.items():
            if state.hits < self.tracker.min_hits:
                continue
            if state.age - state.last_seen >= self.tracker.max_idle:
                continue

            is_fully_occluded = (
                state.track_id not in matched_track_ids
                and state.stereo_detections > 15
                and state.frames_since_last_detection > 0
                and state.frames_since_last_detection <= 20
                and state.last_bbox is not None
            )

            if is_fully_occluded:
                pred_x = float(kf.x[0])
                pred_y = float(kf.x[1])
                pred_z = float(kf.x[2])

                if fx > 0 and pred_z > 0:
                    u = int(pred_x * fx / pred_z + cx)
                    v = int(pred_y * fy / pred_z + cy)

                    last_bbox = state.last_bbox
                    bbox_width = last_bbox[2] - last_bbox[0]
                    bbox_height = last_bbox[3] - last_bbox[1]

                    x1 = max(0, int(u - bbox_width / 2))
                    y1 = max(0, int(v - bbox_height / 2))
                    x2 = int(u + bbox_width / 2)
                    y2 = int(v + bbox_height / 2)

                    filtered_refined_detections.append(
                        {
                            "bbox": np.array([x1, y1, x2, y2], dtype=float),
                            "class": state.cls_id,
                            "score": 0.3,
                            "depth": pred_z,
                            "x": pred_x,
                            "y": pred_y,
                            "z": pred_z,
                            "frame_id": frame_id,
                            "track_id": state.track_id,
                            "predicted": True,
                            "occluded": True,
                            "stereo": False,
                            "established_track": True,
                        }
                    )

        seen_track_ids = set()
        unique_detections = []
        for det in filtered_refined_detections:
            track_id = det.get("track_id")
            if track_id is not None:
                if track_id not in seen_track_ids:
                    seen_track_ids.add(track_id)
                    unique_detections.append(det)
            else:
                det_pos = (
                    round(det.get("x", 0), 1),
                    round(det.get("y", 0), 1),
                    round(det.get("z", 0), 1),
                )
                is_duplicate = False
                for existing in unique_detections:
                    existing_pos = (
                        round(existing.get("x", 0), 1),
                        round(existing.get("y", 0), 1),
                        round(existing.get("z", 0), 1),
                    )
                    if np.linalg.norm(np.array(det_pos) - np.array(existing_pos)) < 0.5:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_detections.append(det)

        filtered_refined_detections = unique_detections

        stereo_points = []
        predicted_points = []

        for det in filtered_refined_detections:
            if det.get("stereo", False):
                stereo_points.append((det["x"], det["y"], det["z"]))
            elif det.get("predicted", False) and det.get("occluded", False):
                predicted_points.append((det["x"], det["y"], det["z"]))

        left_points = stereo_points
        right_points = [(x - self.baseline, y, z) for x, y, z in stereo_points]

        return left_points, right_points, predicted_points, filtered_refined_detections

    def _has_calibration(self) -> bool:
        return (
            self.fx is not None
            and self.fy is not None
            and self.cx is not None
            and self.cy is not None
            and self.baseline > 0.0
        )

    def _collect_frame_names(self, seq_dir: Path) -> List[str]:
        camera_dirs = [
            seq_dir / "image_02" / "data",
            seq_dir / "image_03" / "data",
        ]
        for data_dir in camera_dirs:
            if data_dir.exists():
                return [p.stem for p in sorted(data_dir.glob("*.png"))]
        return []

    def _current_frame_id(self) -> int | None:
        if not self.frame_names:
            return self._frame_idx
        name = self.frame_names[self._frame_idx % len(self.frame_names)]
        try:
            return int(name)
        except ValueError:
            return self._frame_idx

    def _advance_frame(self) -> None:
        if self.frame_names:
            self._frame_idx = (self._frame_idx + 1) % len(self.frame_names)
        else:
            self._frame_idx += 1

    def _load_labels(self, labels_path: Path) -> Dict[int, List[Dict[str, float]]]:
        if not labels_path.exists():
            return {}
        labels: Dict[int, List[Dict[str, float]]] = {}
        with open(labels_path, "r") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                try:
                    frame_id = int(parts[0])
                except ValueError:
                    continue
                bbox = [float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])]
                entry = {
                    "class": parts[2],
                    "track_id": parts[1],
                    "bbox": bbox,
                }
                labels.setdefault(frame_id, []).append(entry)
        return labels
