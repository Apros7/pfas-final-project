from typing import List, Tuple

from detection import Detection


def match_detections(
    left: List[Detection], right: List[Detection], max_vertical_gap: float = 20.0
) -> List[Tuple[Detection, Detection]]:
    matches: List[Tuple[Detection, Detection]] = []
    used: set[int] = set()
    for det_left in left:
        best_idx = -1
        best_cost = float("inf")
        for idx, det_right in enumerate(right):
            if idx in used or det_left.cls_id != det_right.cls_id:
                continue
            disparity = det_left.center[0] - det_right.center[0]
            if disparity <= 0.5:
                continue
            vertical_gap = abs(det_left.center[1] - det_right.center[1])
            if vertical_gap > max_vertical_gap:
                continue
            cost = vertical_gap + abs(disparity) * 0.01
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        if best_idx >= 0:
            used.add(best_idx)
            matches.append((det_left, right[best_idx]))
    return matches

