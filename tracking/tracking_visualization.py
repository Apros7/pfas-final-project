#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from viewer import CameraIntrinsics, Viewer, YOLO_MODEL_PATH

CLASS_NAMES = {0: "Pedestrian", 1: "Cyclist", 2: "Car"}


@dataclass
class Detection:
    bbox: np.ndarray
    cls_id: int
    score: float

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))


class TrackingPainter:
    """Tiny stereo painter that keeps the API approachable."""

    def __init__(
        self,
        model_path: str,
        *,
        conf: float,
        iou: float,
        viewer: Viewer,
        device: str | None = None,
    ) -> None:
        self.model = YOLO(model_path)
        if device:
            self.model.to(device)
        self.conf = conf
        self.iou = iou
        self.left_intr: CameraIntrinsics | None = viewer.cameras[0].intrinsics
        self.right_intr: CameraIntrinsics | None = (
            viewer.cameras[1].intrinsics if len(viewer.cameras) > 1 else None
        )
        self.baseline = viewer.baseline_m

    def __call__(self, left_img: np.ndarray, right_img: np.ndarray):
        left_dets = self._detect(left_img)
        right_dets = self._detect(right_img)
        matches = self._match(left_dets, right_dets)

        left_vis = self._draw(left_img.copy(), left_dets, (0, 170, 255))
        right_vis = self._draw(right_img.copy(), right_dets, (255, 120, 0))
        fusion = self._fusion_view(left_vis, right_vis, matches)
        clouds = self._triangulate(matches)

        return (left_vis, right_vis, fusion), clouds

    def _detect(self, image: np.ndarray) -> List[Detection]:
        results = self.model.predict(image, conf=self.conf, iou=self.iou, verbose=False)
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        return [
            Detection(bbox=xyxy[i], cls_id=int(cls_ids[i]), score=float(scores[i]))
            for i in range(len(cls_ids))
        ]

    def _draw(
        self, image: np.ndarray, detections: Sequence[Detection], color: Tuple[int, int, int]
    ) -> np.ndarray:
        canvas = image
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASS_NAMES.get(det.cls_id, f'id{det.cls_id}')} {det.score:.2f}"
            cv2.putText(
                canvas,
                label,
                (x1 + 4, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _match(
        self, left: Sequence[Detection], right: Sequence[Detection]
    ) -> List[Tuple[Detection, Detection]]:
        matches: List[Tuple[Detection, Detection]] = []
        used: set[int] = set()
        for det_left in left:
            best_idx = -1
            best_score = float("inf")
            for idx, det_right in enumerate(right):
                if idx in used or det_left.cls_id != det_right.cls_id:
                    continue
                disparity = det_left.center[0] - det_right.center[0]
                if disparity <= 1.0:
                    continue
                vertical_gap = abs(det_left.center[1] - det_right.center[1])
                score = vertical_gap + abs(disparity) * 0.02
            if score < best_score:
                best_score = score
                best_idx = idx
            if best_idx >= 0:
                used.add(best_idx)
                matches.append((det_left, right[best_idx]))
        return matches

    def _fusion_view(
        self,
        left_vis: np.ndarray,
        right_vis: np.ndarray,
        matches: Sequence[Tuple[Detection, Detection]],
    ) -> np.ndarray:
        if left_vis.shape != right_vis.shape:
            right_vis = cv2.resize(right_vis, (left_vis.shape[1], left_vis.shape[0]))
        fused = cv2.addWeighted(left_vis, 0.5, right_vis, 0.5, 0)
        for det_left, det_right in matches:
            pt_left = tuple(map(int, det_left.center))
            pt_right = tuple(map(int, det_right.center))
            cv2.line(fused, pt_left, pt_right, (0, 255, 0), 1, cv2.LINE_AA)
        return fused

    def _triangulate(
        self, matches: Sequence[Tuple[Detection, Detection]]
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        if (
            not matches
            or not self.left_intr
            or not self.right_intr
            or not self.baseline
            or self.left_intr.fx == 0
            or self.left_intr.fy == 0
        ):
            return [], []

        left_points: List[Tuple[float, float, float]] = []
        right_points: List[Tuple[float, float, float]] = []
        fx = self.left_intr.fx
        fy = self.left_intr.fy
        cx = self.left_intr.cx
        cy = self.left_intr.cy

        for det_left, det_right in matches:
            disparity = det_left.center[0] - det_right.center[0]
            if disparity <= 1.0:
                continue
            z = fx * self.baseline / disparity
            x = (det_left.center[0] - cx) * z / fx
            y = (det_left.center[1] - cy) * z / fy
            left_points.append((x, y, z))
            right_points.append((x - self.baseline, y, z))
        return left_points, right_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make the stereo tracking demo easy to run.")
    parser.add_argument(
        "--seq-dir",
        type=str,
        default="34759_final_project_rect/seq_01",
        help="Path to the KITTI-style sequence directory",
    )
    parser.add_argument("--model", type=str, default=YOLO_MODEL_PATH, help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--iou", type=float, default=0.35, help="Detection IoU threshold")
    parser.add_argument(
        "--calib",
        type=str,
        default="34759_final_project_rect/calib_cam_to_cam.txt",
        help="Calibration file used to configure the cameras",
    )
    parser.add_argument("--baseline", type=float, default=None, help="Override stereo baseline")
    parser.add_argument("--wait", type=int, default=1, help="Delay (ms) between frames")
    parser.add_argument("--loop", action="store_true", help="Loop sequence forever")
    parser.add_argument("--device", type=str, default=None, help="YOLO device hint (cpu/cuda)")
    parser.add_argument("--map-width", type=int, default=360, help="Top-down map width")
    parser.add_argument("--map-height", type=int, default=1080, help="Top-down map height")
    parser.add_argument("--depth-range", type=float, default=80.0, help="Depth range in meters")
    parser.add_argument(
        "--lateral-range",
        type=float,
        default=30.0,
        help="Lateral range (meters) shown in the map",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer = Viewer(
        args.seq_dir,
        calib_file=args.calib,
        baseline=args.baseline,
        map_size=(args.map_width, args.map_height),
        depth_range=args.depth_range,
        lateral_range=args.lateral_range,
    )
    painter = TrackingPainter(
        args.model,
        conf=args.conf,
        iou=args.iou,
        viewer=viewer,
        device=args.device,
    )
    viewer.run(painter, wait_ms=args.wait, loop=args.loop)


if __name__ == "__main__":
    main()

