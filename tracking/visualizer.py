from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from constants import CLASS_NAMES


def draw_bounding_box(
    image: np.ndarray,
    bbox: np.ndarray,
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        image,
        label,
        (x1 + 3, max(15, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_fusion_view(
    image: np.ndarray, detections: List[Dict[str, float]], frame_id: Optional[int]
) -> np.ndarray:
    overlay = image.copy()

    for det in detections:
        bbox = np.array(det["bbox"], dtype=float)
        class_name = CLASS_NAMES.get(det.get("class", -1), f"id{det.get('class', 0)}")
        track_id = det.get("track_id", None)
        depth = det.get("depth", 0.0)

        if track_id is not None:
            label_text = f"{class_name} #{track_id} {depth:.1f}m"
        else:
            label_text = f"{class_name} {depth:.1f}m"

        is_predicted = det.get("predicted", False)
        is_stereo = det.get("stereo", False)
        is_single_camera = det.get("single_camera", False)
        is_established = det.get("established_track", False)
        is_occluded = det.get("occluded", False)

        if is_predicted and is_occluded and is_established:
            color = (255, 0, 255)
        elif is_stereo:
            color = (0, 255, 0)
        elif is_single_camera and is_established:
            color = (255, 0, 0)
        elif is_single_camera:
            color = (0, 255, 255)
        elif is_predicted:
            color = (255, 255, 255)
        else:
            color = (128, 128, 128)

        draw_bounding_box(overlay, bbox, label_text, color)

    cv2.putText(
        overlay,
        "Fused detections",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def draw_refined_predictions(
    image: np.ndarray, detections: List[Dict[str, float]], frame_id: Optional[int]
) -> np.ndarray:
    overlay = image.copy()

    for det in detections:
        bbox = np.array(det["bbox"], dtype=float)
        class_name = CLASS_NAMES.get(det.get("class", -1), f"id{det.get('class', 0)}")
        track_id = det.get("track_id", None)
        depth = det.get("depth", 0.0)

        if track_id is not None:
            label_text = f"{class_name} #{track_id} {depth:.1f}m"
        else:
            label_text = f"{class_name} {depth:.1f}m"

        is_predicted = det.get("predicted", False)
        is_occluded = det.get("occluded", False)
        is_established = det.get("established_track", False)

        if is_predicted and is_occluded and is_established:
            color = (255, 0, 255)
        elif is_predicted:
            color = (255, 255, 255)
        else:
            color = (0, 255, 0)

        draw_bounding_box(overlay, bbox, label_text, color)

    cv2.putText(
        overlay,
        "Refined detections",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def draw_ground_truth(
    image: np.ndarray, labels: List[Dict[str, float]], frame_id: Optional[int]
) -> np.ndarray:
    for item in labels:
        x1, y1, x2, y2 = map(int, item["bbox"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{item['class']} #{item['track_id']}",
            (x1 + 3, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    header = (
        f"Ground truth frame {frame_id:06d}" if frame_id is not None else "Ground truth"
    )
    cv2.putText(
        image,
        header,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if not labels:
        cv2.putText(
            image,
            "No labels",
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )

    return image


def draw_annotations(
    image: np.ndarray,
    detections: List,
    color: Tuple[int, int, int],
) -> np.ndarray:
    overlay = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
    return overlay

