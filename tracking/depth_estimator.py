from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from detection import Detection


def compute_disparity_map(
    left_img: np.ndarray,
    right_img: np.ndarray,
    sgbm: Optional[object],
) -> Optional[np.ndarray]:
    if sgbm is None:
        return None
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    disparity = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disparity[disparity <= 0.5] = np.nan
    return disparity


def extract_disparity_from_roi(
    disparity_map: np.ndarray, bbox: np.ndarray
) -> Optional[float]:
    x1, y1, x2, y2 = bbox.astype(int)
    roi = disparity_map[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
    if roi.size == 0:
        return None
    valid = roi[np.isfinite(roi)]
    if valid.size == 0:
        return None
    disparity = float(np.median(valid))
    if disparity <= 0.5:
        return None
    return disparity


def triangulate_point(
    u: float,
    v: float,
    disparity: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
) -> Tuple[float, float, float]:
    z = fx * baseline / disparity
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy if fy else 0.0
    return x, y, z


def estimate_depth_from_stereo_match(
    det_left: Detection,
    det_right: Detection,
    disparity_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
) -> Optional[Dict[str, float]]:
    disparity = extract_disparity_from_roi(disparity_map, det_left.bbox)
    if disparity is None:
        return None

    x, y, z = triangulate_point(
        det_left.center[0],
        det_left.center[1],
        disparity,
        fx,
        fy,
        cx,
        cy,
        baseline,
    )

    return {
        "x": x,
        "y": y,
        "z": z,
        "class": det_left.cls_id,
        "stereo": True,
        "bbox": det_left.bbox.copy(),
        "score": det_left.score,
        "depth": z,
    }


def estimate_depth_from_single_camera(
    det: Detection,
    disparity_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
    is_right_camera: bool = False,
) -> Optional[Dict[str, float]]:
    disparity = extract_disparity_from_roi(disparity_map, det.bbox)
    if disparity is None:
        return None

    x, y, z = triangulate_point(
        det.center[0],
        det.center[1],
        disparity,
        fx,
        fy,
        cx,
        cy,
        baseline,
    )

    if is_right_camera:
        x = x + baseline

    return {
        "x": x,
        "y": y,
        "z": z,
        "class": det.cls_id,
        "stereo": False,
        "single_camera": True,
        "bbox": det.bbox.copy(),
        "score": det.score * 0.9,
        "depth": z,
    }

