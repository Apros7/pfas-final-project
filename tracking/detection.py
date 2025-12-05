from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Detection:
    bbox: np.ndarray
    cls_id: int
    score: float
    center: Tuple[float, float]

