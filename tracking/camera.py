from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    tx: float = 0.0


@dataclass(frozen=True)
class CameraModel:
    name: str
    intrinsics: Optional[CameraIntrinsics]
    translation: Optional[np.ndarray] = None


def load_calibration(calib_path: Path) -> dict[str, dict[str, np.ndarray]]:
    if not calib_path or not calib_path.exists():
        return {}

    def parse_numbers(raw_values: str) -> Optional[np.ndarray]:
        cleaned = raw_values.split("#", 1)[0].strip()
        if not cleaned:
            return None
        try:
            data = [float(tok) for tok in cleaned.replace(",", " ").split()]
            return np.asarray(data, dtype=np.float64)
        except ValueError:
            return None

    camera_map: dict[str, dict[str, np.ndarray]] = {}
    with open(calib_path, "r") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or ":" not in raw:
                continue
            key, values = raw.split(":", 1)
            key = key.strip()
            if "_" not in key:
                continue
            field, cam_id = key.rsplit("_", 1)
            numbers = parse_numbers(values)
            if numbers is None:
                continue
            if field.startswith("P"):
                try:
                    numbers = numbers.reshape(3, 4)
                except ValueError:
                    continue
            elif field in {"R", "R_rect", "K"}:
                try:
                    numbers = numbers.reshape(3, 3)
                except ValueError:
                    continue
            camera_map.setdefault(cam_id, {})[field] = numbers
    return camera_map


def extract_intrinsics(entry: dict[str, np.ndarray] | None) -> Optional[CameraIntrinsics]:
    if not entry:
        return None
    matrix = entry.get("P_rect")
    if matrix is None:
        matrix = entry.get("P")
    if matrix is None:
        matrix = entry.get("K")
    if matrix is None:
        return None
    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])
    cx = float(matrix[0, 2])
    cy = float(matrix[1, 2])
    tx = float(matrix[0, 3] / fx) if fx and matrix.shape[1] > 3 else 0.0
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, tx=tx)


def extract_translation(entry: dict[str, np.ndarray] | None) -> Optional[np.ndarray]:
    if not entry:
        return None
    vec = entry.get("T")
    if vec is None:
        return None
    if vec.size >= 3:
        return vec[:3].astype(float)
    return None


def camera_id(camera_name: str) -> str:
    return camera_name.split("_")[-1]


def create_stereo_matcher(
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
    baseline: float,
) -> Optional[object]:
    if fx is None or fy is None or cx is None or cy is None or baseline <= 0.0:
        return None

    import cv2

    num_disparities = 128
    block_size = 5
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=1,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher

