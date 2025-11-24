# stereo_calib.py
import os
from typing import List, Dict, Tuple

import cv2
import numpy as np

from config_calib import SQUARE_SIZE
from mono_calib import load_detections, make_object_points
from config_calib import CALIB_FILE, LEFT_ID, RIGHT_ID
from rect_and_eval import parse_kitti_calib_file  # or copy the function in

def load_gt_intrinsics():
    calib = parse_kitti_calib_file(CALIB_FILE)

    KL_key = f"K_{LEFT_ID}"
    DL_key = f"D_{LEFT_ID}"
    KR_key = f"K_{RIGHT_ID}"
    DR_key = f"D_{RIGHT_ID}"

    assert KL_key in calib and DL_key in calib, f"Missing {KL_key}/{DL_key} in calib file"
    assert KR_key in calib and DR_key in calib, f"Missing {KR_key}/{DR_key} in calib file"

    K_left_gt  = calib[KL_key].reshape(3, 3)
    D_left_gt  = calib[DL_key].reshape(-1, 1)  # shape (5,1)
    K_right_gt = calib[KR_key].reshape(3, 3)
    D_right_gt = calib[DR_key].reshape(-1, 1)

    print("\n[GT] Using ground-truth intrinsics:")
    print("  K_left_gt:\n", K_left_gt)
    print("  D_left_gt:", D_left_gt.ravel())
    print("  K_right_gt:\n", K_right_gt)
    print("  D_right_gt:", D_right_gt.ravel())

    return K_left_gt, D_left_gt, K_right_gt, D_right_gt


# ============================================================
# 1. Build stereo correspondences from detections
# ============================================================

def debug_stereo_correspondences(
    view_idx: int,
    objpoints: List[np.ndarray],
    imgpoints_left: List[np.ndarray],
    imgpoints_right: List[np.ndarray],
    view_meta: List[Dict],
    image_paths_left: List[str],
    image_paths_right: List[str],
    F: np.ndarray = None,
    out_dir: str = "stereo_debug"
):
    """
    Visualize and sanity-check correspondences for a single stereo view.

    - Loads the raw left/right image for the view's image_idx.
    - Draws all chessboard corners with their indices on both images.
    - Stacks images horizontally and draws lines between matching points.
    - If F is given (3x3), prints point-to-epipolar-line errors in the right image.
    """
    os.makedirs(out_dir, exist_ok=True)

    if view_idx < 0 or view_idx >= len(view_meta):
        print(f"[debug] view_idx {view_idx} out of range (0..{len(view_meta)-1})")
        return

    meta = view_meta[view_idx]
    img_idx = int(meta["image_idx"])
    board_name = meta["board_name"]
    pattern_size = meta["pattern_size"]

    pathL = image_paths_left[img_idx]
    pathR = image_paths_right[img_idx]

    imgL = cv2.imread(pathL, cv2.IMREAD_COLOR)
    imgR = cv2.imread(pathR, cv2.IMREAD_COLOR)
    if imgL is None or imgR is None:
        print(f"[debug] Could not load images:\n  {pathL}\n  {pathR}")
        return

    ptsL = imgpoints_left[view_idx].reshape(-1, 2)
    ptsR = imgpoints_right[view_idx].reshape(-1, 2)

    if ptsL.shape != ptsR.shape:
        print(f"[debug] Corner shape mismatch in view {view_idx}")
        return

    n_pts = ptsL.shape[0]

    # Draw corners + indices on copies of the images
    visL = imgL.copy()
    visR = imgR.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (pL, pR) in enumerate(zip(ptsL, ptsR)):
        xL, yL = int(pL[0]), int(pL[1])
        xR, yR = int(pR[0]), int(pR[1])

        # Left: green circle + index
        cv2.circle(visL, (xL, yL), 3, (0, 255, 0), -1)
        cv2.putText(visL, str(i), (xL + 3, yL - 3), font, 0.4, (0, 255, 0), 1)

        # Right: blue circle + index
        cv2.circle(visR, (xR, yR), 3, (255, 0, 0), -1)
        cv2.putText(visR, str(i), (xR + 3, yR - 3), font, 0.4, (255, 0, 0), 1)

    # Stack horizontally and draw lines between corresponding points
    hL, wL = visL.shape[:2]
    hR, wR = visR.shape[:2]
    h = max(hL, hR)
    # pad to same height if needed
    if hL != hR:
        padL = np.zeros((h - hL, wL, 3), dtype=visL.dtype)
        padR = np.zeros((h - hR, wR, 3), dtype=visR.dtype)
        if hL < h:
            visL = np.vstack([visL, padL])
        if hR < h:
            visR = np.vstack([visR, padR])

    combined = np.hstack([visL, visR])

    highlight_indices = {0, 3, 5}

    for i, (pL, pR) in enumerate(zip(ptsL, ptsR)):
        if i not in highlight_indices:
            continue  # skip all other points

        xL, yL = int(pL[0]), int(pL[1])
        xR, yR = int(pR[0]), int(pR[1])
        xR_comb = xR + wL  # shift right x in the combined image

        color = ((37 * i) % 255, (97 * i) % 255, (173 * i) % 255)
        cv2.line(combined, (xL, yL), (xR_comb, yR), color, 3)

    # Save debug image
    out_img = os.path.join(out_dir, f"stereo_corr_view{view_idx:03d}_img{img_idx:02d}_{board_name}.png")
    cv2.imwrite(out_img, combined)
    print(f"[debug] Stereo correspondence visualization saved to:\n  {out_img}")

    # Optional: epipolar error using F (right image distance to epipolar line)
    if F is not None and F.shape == (3, 3):
        ptsL_h = np.hstack([ptsL, np.ones((n_pts, 1))])  # (N,3)
        ptsR_h = np.hstack([ptsR, np.ones((n_pts, 1))])  # (N,3)

        # Epipolar line in right image for each left point: l' = F x
        FxL = (F @ ptsL_h.T).T  # (N,3)
        # Distance from right point to its epipolar line
        num = np.abs(np.sum(ptsR_h * FxL, axis=1))
        denom = np.sqrt(FxL[:, 0]**2 + FxL[:, 1]**2)
        eps = 1e-12
        dist = num / (denom + eps)



def build_stereo_correspondences(
    det_left: List[Dict],
    det_right: List[Dict],
    square_size: float,
):
    """
    Build stereo calibration lists:
      - objpoints: list of (N,3)
      - imgpoints_left: list of (N,2)
      - imgpoints_right: list of (N,2)
      - view_meta: list of dict with info per view

    A 'view' here is a particular (image_idx, board_name) pair that exists
    in BOTH cameras.
    """
    # Index detections by (image_idx, board_name)
    def index_by_key(detections):
        idx: Dict[Tuple[int, str], Dict] = {}
        for det in detections:
            img_idx = int(det["image_idx"])
            board_name = str(det["board_name"])
            key = (img_idx, board_name)
            if key in idx:
                # Should not happen if each board appears once per image,
                # but we warn rather than crash.
                print(f"WARNING: multiple detections for key {key}, "
                      "keeping the last one.")
            idx[key] = det
        return idx

    idxL = index_by_key(det_left)
    idxR = index_by_key(det_right)

    keys_common = sorted(set(idxL.keys()) & set(idxR.keys()))
    print(f"  Found {len(keys_common)} common (image, board) pairs "
          f"between left and right.")

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    view_meta = []

    # Cache object grids per pattern_size
    objp_cache: Dict[Tuple[int, int], np.ndarray] = {}

    for (img_idx, board_name) in keys_common:
        detL = idxL[(img_idx, board_name)]
        detR = idxR[(img_idx, board_name)]

        patternL = tuple(detL["pattern_size"])
        patternR = tuple(detR["pattern_size"])
        if patternL != patternR:
            print(f"  WARNING: pattern mismatch for image {img_idx}, "
                  f"board {board_name}: left={patternL}, right={patternR}. "
                  "Skipping this view.")
            continue

        pattern_size = patternL
        if pattern_size not in objp_cache:
            objp_cache[pattern_size] = make_object_points(
                pattern_size, square_size
            )
        objp = objp_cache[pattern_size]

        cornersL = detL["corners"].astype(np.float32)
        cornersR = detR["corners"].astype(np.float32)

        if cornersL.shape != cornersR.shape:
            print(f"  WARNING: corner count mismatch for image {img_idx}, "
                  f"board {board_name}. Skipping.")
            continue

        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

        view_meta.append({
            "image_idx": img_idx,
            "board_name": board_name,
            "image_name_left": detL["image_name"],
            "image_name_right": detR["image_name"],
            "pattern_size": pattern_size,
        })

    print(f"  Built {len(objpoints)} stereo views.")
    return objpoints, imgpoints_left, imgpoints_right, view_meta



# ============================================================
# 3. Stereo calibration
# ============================================================

def run_stereo_calibration(
    objpoints: List[np.ndarray],
    imgpoints_left: List[np.ndarray],
    imgpoints_right: List[np.ndarray],
    K_left: np.ndarray,
    D_left: np.ndarray,
    K_right: np.ndarray,
    D_right: np.ndarray,
    image_size: Tuple[int, int],
    fix_intrinsics: bool = True,
):
    """
    Run cv2.stereoCalibrate.

    If fix_intrinsics=True, camera matrices and distortions are not changed,
    and only R, T, E, F are estimated.
    """
    flags = 0
    if fix_intrinsics:
        flags |= cv2.CALIB_FIX_INTRINSIC

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100, 1e-5)

    print("  Running cv2.stereoCalibrate...")
    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        K_left.copy(),
        D_left.copy(),
        K_right.copy(),
        D_right.copy(),
        image_size,
        criteria=criteria,
        flags=flags
    )

    print(f"  Stereo RMS reprojection error: {rms:.6f}")
    if fix_intrinsics:
        print("  (Intrinsics were fixed; K1/K2 and D1/D2 are unchanged.)")

    print("  R (rotation from left to right):")
    print(R)
    print("  T (translation from left to right):")
    print(T.ravel())

    return rms, K1, D1, K2, D2, R, T, E, F


# ============================================================
# 4. Full pipeline
# ============================================================

def stereo_pipeline(
    corners_left_npz: str = "corners_left.npz",
    corners_right_npz: str = "corners_right.npz",
    mono_left_npz: str = "mono_left_calib.npz",
    mono_right_npz: str = "mono_right_calib.npz",
):
    print("======= Stereo calibration pipeline =======")

    # 1) Load mono results
    if not os.path.isfile(mono_left_npz) or not os.path.isfile(mono_right_npz):
        raise FileNotFoundError("Mono calibration files not found. "
                                "Run mono_calib.py first.")

    monoL = np.load(mono_left_npz, allow_pickle=True)
    monoR = np.load(mono_right_npz, allow_pickle=True)

    K_left = monoL["K"]
    D_left = monoL["D"]
    K_right = monoR["K"]
    D_right = monoR["D"]
    image_size_arr = monoL["image_size"]
    image_size = (int(image_size_arr[0]), int(image_size_arr[1]))
    print(f"  Image size (raw): {image_size}")

    print("\n[stereo] Estimated intrinsics from mono_calib:")
    print("  K_left_est:\n", K_left)
    print("  D_left_est:", D_left.ravel())
    print("  K_right_est:\n", K_right)
    print("  D_right_est:", D_right.ravel())

    # --- OVERRIDE with ground-truth intrinsics to debug ---
    _, _, _, _ = load_gt_intrinsics()


    # 2) Load detections
    image_paths_left, det_left = load_detections("left", corners_left_npz)
    image_paths_right, det_right = load_detections("right", corners_right_npz)

    # 3) Build stereo correspondences
    objpoints, imgpoints_left, imgpoints_right, view_meta = \
        build_stereo_correspondences(det_left, det_right, SQUARE_SIZE)

    if len(objpoints) == 0:
        raise RuntimeError("No valid stereo views found. "
                           "Check your detections and ROIs.")


    for i in range(13):
        debug_stereo_correspondences(
            view_idx=i,
            objpoints=objpoints,
            imgpoints_left=imgpoints_left,
            imgpoints_right=imgpoints_right,
            view_meta=view_meta,
            image_paths_left=image_paths_left,
            image_paths_right=image_paths_right,
            F=None,     # or None if you don't care about epipolar error
            out_dir="stereo_debug"
    )
    # 5) Stereo calibration with fixed intrinsics
    rms, K1, D1, K2, D2, R, T, E, F = run_stereo_calibration(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        K_left,
        D_left,
        K_right,
        D_right,
        image_size,
        fix_intrinsics=True,
    )

    # 6) Save everything
    out_path = "stereo_calib.npz"
    np.savez_compressed(
        out_path,
        image_size=np.array(image_size),
        K_left=K1,
        D_left=D1,
        K_right=K2,
        D_right=D2,
        R=R,
        T=T,
        E=E,
        F=F,
        rms=rms,
        view_meta=np.array(view_meta, dtype=object),
    )
    print(f"  Saved stereo calibration to {out_path}")
    print("======= Stereo calibration done. =======")


# ============================================================
# 5. Main
# ============================================================

def main():
    stereo_pipeline()


if __name__ == "__main__":
    main()