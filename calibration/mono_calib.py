# mono_calib.py
import os
from typing import List, Dict, Tuple

import cv2
import numpy as np
from rect_and_eval import parse_kitti_calib_file

from config_calib import (
    SQUARE_SIZE,
    CALIB_LEFT_GLOB,
    CALIB_RIGHT_GLOB,
    CALIB_FILE,
    LEFT_ID,
    RIGHT_ID,
)



# ============================================================
# 1. Helpers: load detections, build object points
# ============================================================

def compare_intrinsics_with_gt(cam_name: str, K_est: np.ndarray, D_est: np.ndarray):
    """
    Compare mono-calibrated intrinsics (K_est, D_est) to ground truth
    from calib_cam_to_cam.txt for this camera (left/right).
    """
    calib = parse_kitti_calib_file(CALIB_FILE)

    # Decide which KITTI camera id to use based on cam_name
    if cam_name.lower().startswith("l"):
        cam_id = LEFT_ID   # e.g. "02"
    else:
        cam_id = RIGHT_ID  # e.g. "03"

    key_K = f"K_{cam_id}"
    key_D = f"D_{cam_id}"

    if key_K not in calib or key_D not in calib:
        print(f"[{cam_name}] GT intrinsics not found for {cam_id}; skipping comparison.")
        return

    K_gt = calib[key_K].reshape(3, 3)
    D_gt = calib[key_D].reshape(-1)

    print(f"\n[{cam_name}] Comparison to ground-truth intrinsics (camera {cam_id}):")
    print("  K_est:\n", K_est)
    print("  K_gt:\n", K_gt)
    print("  ΔK = K_est - K_gt:\n", K_est - K_gt)
    print("  max |ΔK|:", np.max(np.abs(K_est - K_gt)))

    print("  D_est:", D_est.ravel())
    print("  D_gt :", D_gt)
    print("  ΔD = D_est - D_gt:", D_est.ravel() - D_gt)
    print("  max |ΔD|:", np.max(np.abs(D_est.ravel() - D_gt)))

def load_detections(cam_name: str, npz_path: str = None):
    """
    Load detections saved by detect_corners.py.

    Returns:
      image_paths: list[str]
      detections: list[dict] with keys:
          'image_idx', 'image_name', 'board_name', 'pattern_size', 'corners'
    """
    if npz_path is None:
        npz_path = f"corners_{cam_name}.npz"

    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"{npz_path} not found. "
                                f"Run detect_corners.py first.")

    data = np.load(npz_path, allow_pickle=True)
    image_paths = list(data["image_paths"])
    detections_array = data["detections"]

    # detections_array is a 1D object array of dicts
    detections: List[Dict] = [det for det in detections_array]

    print(f"[{cam_name}] Loaded {len(detections)} detections "
          f"from {npz_path}")
    return image_paths, detections


def get_image_size_from_paths(image_paths: List[str]) -> Tuple[int, int]:
    """
    Read the first image to get (width, height).
    """
    if not image_paths:
        raise ValueError("No image paths available to determine image size.")
    img = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image {image_paths[0]} "
                      "to determine size.")
    h, w = img.shape[:2]
    return (w, h)


def make_object_points(pattern_size: Tuple[int, int],
                       square_size: float) -> np.ndarray:
    """
    Create 3D object points for a planar chessboard at Z=0.

    pattern_size: (cols, rows) of inner corners
    Returns: (N,3) float32 array.
    """
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    # x = columns (0..cols-1), y = rows (0..rows-1)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp


# ============================================================
# 2. Build calibration dataset and run calibrateCamera
# ============================================================

def build_calib_lists(
    detections: List[Dict],
    square_size: float
):
    """
    From detections, build lists of object_points and image_points, plus
    view metadata.

    Returns:
      objpoints: list of (N,3) float32 arrays
      imgpoints: list of (N,2) float32 arrays
      view_meta: list of dicts with keys:
          'board_name', 'image_idx', 'image_name', 'pattern_size'
    """
    objpoints = []
    imgpoints = []
    view_meta = []

    # Cache object point grids per pattern_size
    objp_cache: Dict[Tuple[int, int], np.ndarray] = {}

    for det in detections:
        pattern_size = tuple(det["pattern_size"])
        if pattern_size not in objp_cache:
            objp_cache[pattern_size] = make_object_points(
                pattern_size, square_size
            )

        objpoints.append(objp_cache[pattern_size])
        imgpoints.append(det["corners"].astype(np.float32))

        view_meta.append({
            "board_name": det["board_name"],
            "image_idx": int(det["image_idx"]),
            "image_name": det["image_name"],
            "pattern_size": pattern_size,
        })

    print(f"  Built {len(objpoints)} calibration views")
    return objpoints, imgpoints, view_meta


def run_calibration(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    image_size: Tuple[int, int],
):
    """
    Mono calibration using ONLY cv2.CALIB_FIX_ASPECT_RATIO.
    No explicit intrinsic guess is provided.

    OpenCV:
      - builds its own initial guess for K and D,
      - keeps fx/fy fixed to that internal guess (aspect ratio),
      - optimizes everything else normally.
    """
    flags = cv2.CALIB_FIX_ASPECT_RATIO   # 👈 only this flag

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100, 1e-6)

    print("  Running cv2.calibrateCamera...")
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,      # no intrinsic guess
        None,      # no distortion guess
        flags=flags,
        criteria=criteria,
    )

    print(f"  Global RMS reprojection error: {rms:.6f}")
    print("  K:\n", K)
    print("  D:", D.ravel())

    # Per-view RMS
    per_view_rms = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
        proj = proj.reshape(-1, 2)
        err = imgp.reshape(-1, 2) - proj
        err_sq = np.sum(err**2, axis=1)
        per_view_rms.append(float(np.sqrt(np.mean(err_sq))))

    print("  Per-view RMS stats:")
    print(f"    min:    {min(per_view_rms):.6f}")
    print(f"    max:    {max(per_view_rms):.6f}")
    print(f"    mean:   {np.mean(per_view_rms):.6f}")
    print(f"    median: {np.median(per_view_rms):.6f}")

    return rms, K, D, rvecs, tvecs, per_view_rms


# ============================================================
# 4. Full pipeline for one camera
# ============================================================

def calibrate_single_camera(
    cam_name: str,
    corners_npz_path: str = None,
    max_views_per_board: int = 3
):
    """
    Full mono calibration pipeline for one camera:
      1) Load detections
      2) Build calib lists
      3) Initial calibration
      4) Select best views per board
      5) Final calibration
      6) Save results to mono_<cam_name>_calib.npz
    """
    print(f"\n======= Mono calibration for {cam_name.upper()} camera =======")

    image_paths, detections = load_detections(cam_name, corners_npz_path)
    image_size = get_image_size_from_paths(image_paths)
    print(f"  Image size: {image_size}")

    # Step 1: build full dataset
    objpoints, imgpoints, view_meta = build_calib_lists(
        detections, SQUARE_SIZE
    )

    # Step 2: initial calibration
    rms, K, D, rvecs, tvecs, per_view_rms = run_calibration(
        objpoints, imgpoints, image_size
    )


    # Print camera matrix and distortion
    print(f"\n[{cam_name}] Final RMS: {rms:.6f}")
    print(f"[{cam_name}] Camera matrix K:\n{K}")
    print(f"[{cam_name}] Distortion coefficients D:\n{D.ravel()}")

    compare_intrinsics_with_gt(cam_name, K, D)
    # Step 5: save results
    out_path = f"mono_{cam_name}_calib.npz"
    np.savez_compressed(
        out_path,
        image_size=np.array(image_size),
        K=K,
        D=D,
        rms=rms,
        per_view_rms=np.array(per_view_rms),
        view_meta=np.array(view_meta, dtype=object),
    )
    print(f"[{cam_name}] Saved calibration to {out_path}")

    return {
        "image_size": image_size,
        "K": K,
        "D": D,
        "rms": rms,
        "per_view_rms": per_view_rms,
        "view_meta": view_meta,
    }


# ============================================================
# 5. Main
# ============================================================

def main():
    # LEFT camera
    left_result = calibrate_single_camera("left")

    # RIGHT camera
    right_result = calibrate_single_camera("right")

    print("\nAll mono calibrations done.")


if __name__ == "__main__":
    main()
