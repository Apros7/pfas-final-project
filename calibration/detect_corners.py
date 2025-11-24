# detect_corners.py
import glob
import os
import shutil
from typing import List, Dict, Tuple

import cv2
import numpy as np

from config_calib import (
    DEBUG_DIR,
    CALIB_LEFT_GLOB,
    CALIB_RIGHT_GLOB,
    BOARDS,
)

if os.path.exists(DEBUG_DIR):
    shutil.rmtree(DEBUG_DIR)

os.makedirs(DEBUG_DIR, exist_ok=True)

# ============================================================
# 1. Helpers: grids, RANSAC checks, drawing
# ============================================================

def make_board_grid(pattern_size: Tuple[int, int]) -> np.ndarray:
    """
    Create canonical 2D grid coordinates for a chessboard in board coordinates.
    shape: (N, 2), where N = cols*rows.
    """
    cols, rows = pattern_size
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    return grid


def ransac_check_board(grid_xy: np.ndarray,
                       corners_xy: np.ndarray,
                       reproj_thresh: float = 2.0,
                       min_inlier_ratio: float = 0.9):
    """
    Detects bad boards (points not on the corners)

    Fit homography from board grid -> image corners using RANSAC.
    Returns (ok, H, mask) where:
      - ok is True if inlier ratio >= min_inlier_ratio and H not None
      - H is 3x3 homography or None
      - mask is Nx1 uint8 inlier mask or None
    """
    assert grid_xy.shape == corners_xy.shape
    if grid_xy.shape[0] < 4:
        return False, None, None

    H, mask = cv2.findHomography(
        grid_xy, corners_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_thresh
    )
    if H is None or mask is None:
        return False, None, None

    inlier_ratio = float(mask.sum()) / float(mask.size)
    ok = inlier_ratio >= min_inlier_ratio
    return ok, H, mask


def draw_corners_debug(
    img_color: np.ndarray,
    corners_xy: np.ndarray,
    pattern_size: Tuple[int, int],
    board_name: str,
    out_path: str,
    ransac_mask: np.ndarray = None,  # kept for API compatibility, not used
):
    """
    Draw detected corners using OpenCV's drawChessboardCorners and save to out_path.
    """
    vis = img_color.copy()

    # OpenCV expects corners as (N, 1, 2) float32 in the same coordinate system
    corners_for_draw = corners_xy.reshape(-1, 1, 2).astype(np.float32)

    # patternWasFound=True because corners come from a successful detection
    cv2.drawChessboardCorners(vis, pattern_size, corners_for_draw, True)

    cv2.putText(vis, board_name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)


# ============================================================
# 2. Core: detect chessboard in ROI for one board & image
# ============================================================

def detect_board_in_roi(
    gray: np.ndarray,
    roi: Tuple[int, int, int, int],
    pattern_size: Tuple[int, int]
) -> np.ndarray:
    """
    Try to detect chessboard corners for a single board inside a given ROI.
    Returns corners as (N,2) in FULL image coordinates, or None on failure.
    """
    x, y, w, h = roi
    roi_gray = gray[y:y + h, x:x + w]
    ret, corners = cv2.findChessboardCornersSB(roi_gray, pattern_size)

    if not ret or corners is None:
        return None

    corners = corners.astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30, 0.01)
    
    cv2.cornerSubPix(
        roi_gray,
        corners,
        winSize=(3, 3),
        zeroZone=(-1, -1),
        criteria=criteria
    )

    corners_full = corners.reshape(-1, 2)
    corners_full[:, 0] += x
    corners_full[:, 1] += y

    return corners_full


# ============================================================
# 3. Pipeline for one camera
# ============================================================

def detect_corners_for_camera(
    cam_name: str,
    image_glob: str,
    boards: List,
    debug_root: str = DEBUG_DIR,
    ransac_thresh: float = 2.0,
    min_inlier_ratio: float = 0.9,
):
    """
    Run detection for all images for one camera.

    Returns a dictionary with:
      - image_paths: list[str]
      - detections: list[dict] with keys:
            'image_idx', 'image_name', 'board_name', 'pattern_size',
            'corners' (Nx2 float32)
    """
    image_paths = sorted(glob.glob(image_glob))[:4] #ONLY USING 4 FIRST IMAGES, BECAUSE ITSFASTER
    print(f"[{cam_name}] Found {len(image_paths)} images")
    all_detections: List[Dict] = []

    for img_idx, img_path in enumerate(image_paths):
        img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_color is None:
            print(f"[{cam_name}] WARNING: Could not read {img_path}")
            continue

        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        print(f"[{cam_name}] Processing image {img_idx}: {img_path}")

        for board in boards:
            roi = board.roi_left if cam_name == "left" else board.roi_right

            corners_full = detect_board_in_roi(
                gray, roi, board.pattern_size
            )
            if corners_full is None:
                print(f"  - {board.name}: detection FAILED")
                continue

            grid = make_board_grid(board.pattern_size)
            ok, H, mask = ransac_check_board(
                grid, corners_full,
                reproj_thresh=ransac_thresh,
                min_inlier_ratio=min_inlier_ratio
            )
            if not ok:
                print(f"  - {board.name}: RANSAC REJECTED (too many outliers)")
                dbg_path = os.path.join(
                    debug_root,
                    f"{cam_name}_bad",
                    f"img{img_idx:02d}_{board.name}.png"
                )
                draw_corners_debug(
                    img_color, corners_full, board.pattern_size,
                    board.name, dbg_path, ransac_mask=mask
                )
                continue

            print(f"  - {board.name}: OK "
                  f"(corners={len(corners_full)}, "
                  f"inliers={int(mask.sum())}/{mask.size})")

            dbg_path = os.path.join(
                debug_root,
                f"{cam_name}_ok",
                f"img{img_idx:02d}_{board.name}.png"
            )
            draw_corners_debug(
                img_color, corners_full, board.pattern_size,
                board.name, dbg_path, ransac_mask=mask
            )

            det = {
                "image_idx": img_idx,
                "image_name": os.path.basename(img_path),
                "board_name": board.name,
                "pattern_size": board.pattern_size,
                "corners": corners_full.astype(np.float32),
            }
            all_detections.append(det)

    out_path = f"corners_{cam_name}.npz"
    print(f"[{cam_name}] Saving {len(all_detections)} detections to {out_path}")

    image_names = [os.path.basename(p) for p in image_paths]
    np.savez_compressed(
        out_path,
        image_paths=np.array(image_paths, dtype=object),
        image_names=np.array(image_names, dtype=object),
        detections=np.array(all_detections, dtype=object),
    )

    return {
        "image_paths": image_paths,
        "detections": all_detections,
    }


# ============================================================
# 4. Main
# ============================================================

def main():
    detect_corners_for_camera(
        cam_name="left",
        image_glob=CALIB_LEFT_GLOB,
        boards=BOARDS,
    )

    detect_corners_for_camera(
        cam_name="right",
        image_glob=CALIB_RIGHT_GLOB,
        boards=BOARDS,
    )

    print("[DONE] Corner detection completed for both cameras.")


if __name__ == "__main__":
    main()
