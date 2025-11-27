# rectify_and_eval.py
import os
import glob
from typing import Tuple, Dict, List

import cv2
import numpy as np

from config_calib import (
    CALIB_LEFT_GLOB,
    CALIB_RIGHT_GLOB,
    CALIB_FILE,
    DEBUG_DIR,
    LEFT_ID,
    RIGHT_ID,
)

# ======================================================================
# 0. Paths / config for this script
# ======================================================================

RAW_LEFT_GLOB  = CALIB_LEFT_GLOB
RAW_RIGHT_GLOB = CALIB_RIGHT_GLOB

# Provided rectified images (adjust if needed)
RECT_LEFT_GLOB_PROVIDED  = "p_calibration/34759_final_project_rect/image_02/data/*.png"
RECT_RIGHT_GLOB_PROVIDED = "p_calibration/34759_final_project_rect/image_03/data/*.png"

OUT_RECT_DIR_LEFT   = "rectified_my/image_02"
OUT_RECT_DIR_RIGHT  = "rectified_my/image_03"
OUT_EPI_DIR         = "rectified_my_debug/epipolar"
OUT_DISP_DIR        = "rectified_my_debug/disparity"

os.makedirs(OUT_RECT_DIR_LEFT, exist_ok=True)
os.makedirs(OUT_RECT_DIR_RIGHT, exist_ok=True)
os.makedirs(OUT_EPI_DIR, exist_ok=True)
os.makedirs(OUT_DISP_DIR, exist_ok=True)


# ======================================================================
# 1. Load stereo calibration + calib_cam_to_cam.txt
# ======================================================================

def load_gt_stereo_from_kitti():
    calib = parse_kitti_calib_file(CALIB_FILE)

    KL_key, DL_key = f"K_{LEFT_ID}",  f"D_{LEFT_ID}"
    KR_key, DR_key = f"K_{RIGHT_ID}", f"D_{RIGHT_ID}"
    RL_key, TL_key = f"R_{LEFT_ID}",  f"T_{LEFT_ID}"
    RR_key, TR_key = f"R_{RIGHT_ID}", f"T_{RIGHT_ID}"
    S_key          = f"S_{LEFT_ID}"

    K_left  = calib[KL_key].reshape(3, 3)
    D_left  = calib[DL_key].reshape(-1, 1)
    K_right = calib[KR_key].reshape(3, 3)
    D_right = calib[DR_key].reshape(-1, 1)
    R_left  = calib[RL_key].reshape(3, 3)
    T_left  = calib[TL_key].reshape(3)
    R_right = calib[RR_key].reshape(3, 3)
    T_right = calib[TR_key].reshape(3)
    S       = calib[S_key].reshape(-1)
    raw_image_size = (int(S[0]), int(S[1]))  # (width, height)

    # Relative transform from left to right
    R_rel = R_right @ R_left.T
    T_rel = T_right - R_rel @ T_left

    print("\n[GT] Loaded full stereo calib from KITTI file.")
    print("  raw_image_size:", raw_image_size)
    print("  baseline ||T_rel||:", np.linalg.norm(T_rel))

    return K_left, D_left, K_right, D_right, R_rel, T_rel, raw_image_size


def load_stereo_calib(path: str = "stereo_calib.npz"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found. Run stereo_calib.py first.")
    data = np.load(path, allow_pickle=True)
    K_left  = data["K_left"]
    D_left  = data["D_left"]
    K_right = data["K_right"]
    D_right = data["D_right"]
    R       = data["R"]
    T       = data["T"]
    image_size_arr = data["image_size"]
    # This is the RAW image size used for calibration
    raw_image_size = (int(image_size_arr[0]), int(image_size_arr[1]))
    print("[stereo] Loaded stereo calibration from", path)
    print("  raw_image_size:", raw_image_size)
    return K_left, D_left, K_right, D_right, R, T, raw_image_size


def parse_kitti_calib_file(path: str) -> Dict[str, np.ndarray]:
    """
    Parse calib_cam_to_cam.txt style file into a dict:
      key -> np.ndarray of floats

    Handles lines like:
      S_02: <2 floats>
      K_02: <9 floats>
      D_02: <5 floats>
      R_02: <9 floats>
      T_02: <3 floats>
      S_rect_02: <2 floats>
      R_rect_02: <9 floats>
      P_rect_02: <12 floats>
    """
    if not os.path.isfile(path):
        print(f"[kitti] WARNING: calib file {path} not found.")
        return {}

    calib = {}
    with open(path, "r") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, vals = line.split(":", 1)
            key = key.strip()
            vals = vals.strip()
            if not vals:
                continue
            parts = vals.split()
            floats = [float(v) for v in parts]

            arr = np.array(floats, dtype=np.float64)
            if len(floats) == 12:
                arr = arr.reshape(3, 4)
            elif len(floats) == 9:
                arr = arr.reshape(3, 3)
            # lengths 2 (S_*), 3 (T_*), 5 (D_*), 1 (corner_dist) stay 1D

            calib[key] = arr

    print(f"[kitti] Parsed {len(calib)} entries from {path}")
    return calib


# ======================================================================
# 2. Comparisons with provided calibration
# ======================================================================

def compare_intrinsics_with_provided(
    K_left_est, D_left_est,
    K_right_est, D_right_est,
    calib: Dict[str, np.ndarray],
):
    key_KL = f"K_{LEFT_ID}"
    key_DL = f"D_{LEFT_ID}"
    key_KR = f"K_{RIGHT_ID}"
    key_DR = f"D_{RIGHT_ID}"

    if key_KL not in calib or key_DL not in calib or \
       key_KR not in calib or key_DR not in calib:
        print("[compare] Intrinsic keys not all present; skipping intrinsic comparison.")
        return

    K_left_gt  = calib[key_KL].reshape(3, 3)
    D_left_gt  = calib[key_DL].reshape(-1)
    K_right_gt = calib[key_KR].reshape(3, 3)
    D_right_gt = calib[key_DR].reshape(-1)

    print("\n[compare] Intrinsics comparison:")
    print("  Left camera:")
    print("    K_est:\n", K_left_est)
    print("    K_gt:\n", K_left_gt)
    print("    max |ΔK|:", np.max(np.abs(K_left_est - K_left_gt)))
    print("    D_est:", D_left_est.ravel())
    print("    D_gt :", D_left_gt)
    print("    max |ΔD|:", np.max(np.abs(D_left_est.ravel() - D_left_gt)))

    print("  Right camera:")
    print("    K_est:\n", K_right_est)
    print("    K_gt:\n", K_right_gt)
    print("    max |ΔK|:", np.max(np.abs(K_right_est - K_right_gt)))
    print("    D_est:", D_right_est.ravel())
    print("    D_gt :", D_right_gt)
    print("    max |ΔD|:", np.max(np.abs(D_right_est.ravel() - D_right_gt)))


def compare_with_provided_rectification(
    P1: np.ndarray,
    P2: np.ndarray,
    calib: Dict[str, np.ndarray],
):
    key_PL = f"P_rect_{LEFT_ID}"
    key_PR = f"P_rect_{RIGHT_ID}"

    if key_PL not in calib or key_PR not in calib:
        print(f"[compare] Provided P_rect matrices not found ({key_PL}/{key_PR}). Skipping comparison.")
        return

    P_gt_left = calib[key_PL]
    P_gt_right = calib[key_PR]

    if P_gt_left.shape != (3, 4) or P_gt_right.shape != (3, 4):
        print("[compare] Provided P_rect have unexpected shape. Skipping.")
        return

    print("\n[compare] Rectification projection matrices:")
    print("  Our P1 vs provided", key_PL)
    print("    max |Δ|:", np.max(np.abs(P1 - P_gt_left)))
    print("  Our P2 vs provided", key_PR)
    print("    max |Δ|:", np.max(np.abs(P2 - P_gt_right)))

    # Baseline from P (approx)
    fx_gt = P_gt_left[0, 0]
    B_gt = -P_gt_right[0, 3] / fx_gt if fx_gt != 0 else np.nan

    fx_ours = P1[0, 0]
    B_ours = -P2[0, 3] / fx_ours if fx_ours != 0 else np.nan

    print(f"  Baseline (from P):  ours={B_ours:.6f}  provided={B_gt:.6f}")


def compare_with_provided_extrinsics(
    R_est: np.ndarray,
    T_est: np.ndarray,
    calib: Dict[str, np.ndarray],
):
    key_RL = f"R_{LEFT_ID}"
    key_TL = f"T_{LEFT_ID}"
    key_RR = f"R_{RIGHT_ID}"
    key_TR = f"T_{RIGHT_ID}"

    needed = {key_RL, key_TL, key_RR, key_TR}
    if not needed.issubset(calib.keys()):
        print("[compare] R_i/T_i not all present; skipping extrinsic comparison.")
        return

    R2 = calib[key_RL].reshape(3, 3)
    T2 = calib[key_TL].reshape(3)
    R3 = calib[key_RR].reshape(3, 3)
    T3 = calib[key_TR].reshape(3)

    # Relative transform: from cam_left to cam_right.
    # If cams are wrt same reference: X_i = R_i * X_ref + T_i
    # => X_r = Rr Rl^T X_l + (Tr - Rr Rl^T Tl)
    R_rel_gt = R3 @ R2.T
    T_rel_gt = T3 - R_rel_gt @ T2

    R_delta = R_rel_gt @ R_est.T
    angle = np.arccos(np.clip((np.trace(R_delta) - 1) / 2.0, -1.0, 1.0))
    angle_deg = np.degrees(angle)

    B_gt = np.linalg.norm(T_rel_gt)
    B_est = np.linalg.norm(T_est)

    print("\n[compare] Extrinsics comparison (left=cam", LEFT_ID, ", right=cam", RIGHT_ID, "):")
    print(f"  Baseline:  ours={B_est:.6f}  provided={B_gt:.6f}")
    print(f"  Rotation difference angle: {angle_deg:.6f} deg")


# ======================================================================
# 3. Rectification + maps
# ======================================================================

def compute_rectification(
    K_left, D_left, K_right, D_right, R, T,
    raw_image_size: Tuple[int, int],
    rect_image_size: Tuple[int, int] = None,
    alpha: float = 0.0,
    zero_disparity: bool = True,
):
    """
    Run cv2.stereoRectify.

    raw_image_size: size of input images used for calibration (width, height)
    rect_image_size: desired size of rectified images (width, height), e.g. S_rect_02.
    """
    flags = 0
    if zero_disparity:
        flags |= cv2.CALIB_ZERO_DISPARITY

    if rect_image_size is None:
        rect_image_size = raw_image_size

    print("\n[rectify] Running cv2.stereoRectify...")
    R1, R2, P1, P2, Q, valid_roi1, valid_roi2 = cv2.stereoRectify(
        K_left, D_left,
        K_right, D_right,
        raw_image_size,
        R, T,
        flags=flags,
        alpha=alpha,
        newImageSize=rect_image_size,
    )

    print("[rectify] raw_image_size:", raw_image_size)
    print("[rectify] rect_image_size:", rect_image_size)
    print("[rectify] R1:\n", R1)
    print("[rectify] R2:\n", R2)
    print("[rectify] P1:\n", P1)
    print("[rectify] P2:\n", P2)
    print("[rectify] valid ROI1:", valid_roi1)
    print("[rectify] valid ROI2:", valid_roi2)

    return R1, R2, P1, P2, Q, valid_roi1, valid_roi2, rect_image_size


def build_rectify_maps(
    K_left, D_left, K_right, D_right,
    R1, R2, P1, P2,
    rect_image_size: Tuple[int, int],
):
    w, h = rect_image_size

    print("\n[rectify] Building undistort-rectify maps "
          f"for rect_image_size={rect_image_size}...")
    map1L, map2L = cv2.initUndistortRectifyMap(
        K_left, D_left,
        R1, P1,
        (w, h),
        m1type=cv2.CV_32FC1,
    )
    map1R, map2R = cv2.initUndistortRectifyMap(
        K_right, D_right,
        R2, P2,
        (w, h),
        m1type=cv2.CV_32FC1,
    )

    print("  map1L x-range:", float(np.min(map1L)), "to", float(np.max(map1L)))
    print("  map2L y-range:", float(np.min(map2L)), "to", float(np.max(map2L)))
    print("  map1R x-range:", float(np.min(map1R)), "to", float(np.max(map1R)))
    print("  map2R y-range:", float(np.min(map2R)), "to", float(np.max(map2R)))

    return map1L, map2L, map1R, map2R

# ======================================================================
# 4. Apply rectification to raw image pairs
# ======================================================================

def load_raw_pairs():
    left_paths = sorted(glob.glob(RAW_LEFT_GLOB))
    right_paths = sorted(glob.glob(RAW_RIGHT_GLOB))

    if len(left_paths) == 0 or len(right_paths) == 0:
        raise RuntimeError("No raw images found. Check RAW_LEFT_GLOB/RAW_RIGHT_GLOB.")

    if len(left_paths) != len(right_paths):
        print("[warn] Raw left/right counts differ, truncating to min length.")
        n = min(len(left_paths), len(right_paths))
        left_paths = left_paths[:n]
        right_paths = right_paths[:n]

    print(f"[raw] Found {len(left_paths)} raw stereo pairs.")
    return left_paths, right_paths


def rectify_and_save_pairs(
    map1L, map2L, map1R, map2R,
    sample_indices: List[int] = None,
):
    left_paths, right_paths = load_raw_pairs()
    n_pairs = len(left_paths)

    if sample_indices is None:
        sample_indices = list(range(min(5, n_pairs)))

    print("[rectify] Rectifying pairs with indices:", sample_indices)

    for idx in sample_indices:
        if idx < 0 or idx >= n_pairs:
            continue

        pathL = left_paths[idx]
        pathR = right_paths[idx]

        imgL = cv2.imread(pathL, cv2.IMREAD_COLOR)
        imgR = cv2.imread(pathR, cv2.IMREAD_COLOR)
        if imgL is None or imgR is None:
            print(f"  Pair {idx}: failed to load images.")
            continue

        rectL = cv2.remap(imgL, map1L, map2L, interpolation=cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, map1R, map2R, interpolation=cv2.INTER_LINEAR)

        baseL = os.path.basename(pathL)
        baseR = os.path.basename(pathR)

        outL = os.path.join(OUT_RECT_DIR_LEFT,  f"rect_{idx:03d}_" + baseL)
        outR = os.path.join(OUT_RECT_DIR_RIGHT, f"rect_{idx:03d}_" + baseR)

        cv2.imwrite(outL, rectL)
        cv2.imwrite(outR, rectR)

        print(f"  Saved rectified pair {idx}:")

        save_epipolar_debug(rectL, rectR, idx)
        save_disparity_map(rectL, rectR, idx)


# ======================================================================
# 5. Debug: epipolar lines & disparity
# ======================================================================

def save_epipolar_debug(rectL: np.ndarray, rectR: np.ndarray, idx: int):
    h, w = rectL.shape[:2]
    combined = np.hstack([rectL, rectR])

    for y in range(0, h, 40):
        cv2.line(combined, (0, y), (2 * w - 1, y), (0, 255, 0), 1)

    out_path = os.path.join(OUT_EPI_DIR, f"epi_{idx:03d}.png")
    cv2.imwrite(out_path, combined)
    print(f"    Epipolar debug saved to {out_path}")


def save_disparity_map(rectL: np.ndarray, rectR: np.ndarray, idx: int):
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    min_disp = 0
    num_disp = 128  # must be multiple of 16
    block_size = 5

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    # Normalize to 0..255 for visualization
    disp_norm = cv2.normalize(disp, None, alpha=0, beta=255,
                              norm_type=cv2.NORM_MINMAX)
    disp_vis = disp_norm.astype(np.uint8)

    out_path = os.path.join(OUT_DISP_DIR, f"disp_{idx:03d}.png")
    cv2.imwrite(out_path, disp_vis)
    print(f"    Disparity map saved to {out_path}")


# ======================================================================
# 7. Main pipeline
# ======================================================================

def main():
    # 1) Load our stereo calibration
    K_left, D_left, K_right, D_right, R, T, raw_image_size = load_stereo_calib()
    #K_left, D_left, K_right, D_right, R, T, raw_image_size = load_gt_stereo_from_kitti()

    # 2) Load calib_cam_to_cam.txt and get rectified size for this stereo pair
    calib_dict = parse_kitti_calib_file(CALIB_FILE)

    key_Srect = f"S_rect_{LEFT_ID}"
    if key_Srect in calib_dict:
        arr = calib_dict[key_Srect].reshape(-1)
        rect_image_size = (int(arr[0]), int(arr[1]))
    else:
        print(f"[rectify] {key_Srect} not found; using raw_image_size.")
        rect_image_size = raw_image_size

    # 3) Compare intrinsics to provided calibration
    compare_intrinsics_with_provided(
        K_left, D_left,
        K_right, D_right,
        calib_dict,
    )

    # 4) Compute rectification (match S_rect_* from file)
    R1, R2, P1, P2, Q, roi1, roi2, rect_image_size = compute_rectification(
        K_left, D_left, K_right, D_right,
        R, T,
        raw_image_size=raw_image_size,
        rect_image_size=rect_image_size,
        alpha=0,
        zero_disparity=True,
    )

    # 5) Compare our rectification P1/P2 and extrinsics to provided
    compare_with_provided_rectification(P1, P2, calib_dict)
    compare_with_provided_extrinsics(R, T, calib_dict)

    # 6) Build rectify maps for rectified resolution
    map1L, map2L, map1R, map2R = build_rectify_maps(
        K_left, D_left, K_right, D_right,
        R1, R2, P1, P2,
        rect_image_size,
    )

    # 7) Rectify some raw pairs and produce debug outputs
    rectify_and_save_pairs(map1L, map2L, map1R, map2R,
                           sample_indices=None)


if __name__ == "__main__":
    main()
