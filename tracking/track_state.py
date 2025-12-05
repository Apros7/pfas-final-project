from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TrackState:
    track_id: int
    cls_id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    last_seen: int
    hits: int
    age: int
    stereo_detections: int = 0
    single_camera_frames: int = 0
    last_detection_type: str = "stereo"
    frames_since_last_detection: int = 0
    last_bbox: Optional[np.ndarray] = None
    position_history: list = None

