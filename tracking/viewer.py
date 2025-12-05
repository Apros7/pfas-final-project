from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import cv2
import numpy as np

from camera import (
    CameraModel,
    CameraIntrinsics,
    load_calibration,
    extract_intrinsics,
    extract_translation,
    camera_id,
)
from types import PaintReturn


@dataclass(frozen=True)
class StereoFrame:
    name: str
    left: np.ndarray
    right: np.ndarray


class Viewer:
    def __init__(
        self,
        seq_dir: str | Path,
        *,
        camera_left: str = "image_02",
        camera_right: str = "image_03",
        calib_file: str | Path | None = "34759_final_project_rect/calib_cam_to_cam.txt",
        baseline: float | None = None,
        map_size: Tuple[int, int] = (360, 1080),
        depth_range: float = 80.0,
        lateral_range: float = 30.0,
        window_name: str = "PFAS Viewer",
    ):
        self.seq_dir = Path(seq_dir)
        self.window_name = window_name
        self.map_size = map_size
        self.depth_range = depth_range
        self.lateral_range = lateral_range

        camera_names = [camera_left, camera_right]
        calib_map = load_calibration(Path(calib_file)) if calib_file else {}
        self.cameras: List[CameraModel] = []
        for name in camera_names:
            cam_entry = calib_map.get(camera_id(name))
            self.cameras.append(
                CameraModel(
                    name=name,
                    intrinsics=extract_intrinsics(cam_entry),
                    translation=extract_translation(cam_entry),
                )
            )

        self.frames = self._load_frames(camera_names)
        if not self.frames:
            raise RuntimeError(f"No frames found under {self.seq_dir}")

        self.baseline_m = self._resolve_baseline(baseline)

    def _load_frames(self, camera_names: Sequence[str]) -> List[StereoFrame]:
        left_dir = self.seq_dir / camera_names[0] / "data"
        right_dir = self.seq_dir / camera_names[1] / "data"
        left_files = sorted(left_dir.glob("*.png"))
        frames: List[StereoFrame] = []
        for left_path in left_files:
            right_path = right_dir / left_path.name
            if not right_path.exists():
                continue
            left_img = cv2.imread(str(left_path))
            right_img = cv2.imread(str(right_path))
            if left_img is None or right_img is None:
                continue
            if left_img.shape != right_img.shape:
                right_img = cv2.resize(
                    right_img, (left_img.shape[1], left_img.shape[0])
                )
            frames.append(
                StereoFrame(
                    name=left_path.stem,
                    left=left_img,
                    right=right_img,
                )
            )
        return frames

    def _resolve_baseline(self, user_value: float | None) -> float:
        if user_value:
            return float(user_value)
        if len(self.cameras) < 2:
            return 0.0
        left, right = self.cameras[0], self.cameras[1]
        if left.translation is not None and right.translation is not None:
            delta = np.linalg.norm(left.translation - right.translation)
            if delta > 0:
                return float(delta)
        if left.intrinsics and right.intrinsics:
            delta = abs(right.intrinsics.tx - left.intrinsics.tx)
            if delta > 0:
                return float(delta)
        return 0.54

    def run(
        self,
        painter: Callable[[np.ndarray, np.ndarray], PaintReturn],
        *,
        wait_ms: int = 1,
        loop: bool = False,
        save_path: str | Path | None = None,
    ) -> None:
        idx = 0
        total = len(self.frames)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        video_writer: cv2.VideoWriter | None = None
        target_path: Path | None = None
        if save_path:
            target_path = Path(save_path).expanduser()
            target_path.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1.0, 1000.0 / max(1, wait_ms))
        while idx < total:
            frame = self.frames[idx]
            painted, clouds = painter(frame.left.copy(), frame.right.copy())
            if not painted:
                raise ValueError("Painter must return at least one image view")
            stacked = self._stack_images(painted)
            map_panel = self._map_from_points(clouds)
            if map_panel.shape[0] != stacked.shape[0]:
                map_panel = cv2.resize(
                    map_panel, (map_panel.shape[1], stacked.shape[0])
                )
            composite = np.concatenate([stacked, map_panel], axis=1)
            if target_path and video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                height, width = composite.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(target_path),
                    fourcc,
                    15,
                    (width, height),
                )
            if video_writer:
                video_writer.write(composite)
            cv2.imshow(self.window_name, composite)
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                key = cv2.waitKey(-1) & 0xFF
                if key in (27, ord("q")):
                    break
            idx += 1
            if idx >= total and loop:
                idx = 0
        if video_writer:
            video_writer.release()
        cv2.destroyWindow(self.window_name)

    def _stack_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        reference = images[0].shape[1], images[0].shape[0]
        resized = [
            cv2.resize(img, reference)
            if img.shape[:2] != images[0].shape[:2]
            else img
            for img in images
        ]
        return np.concatenate(resized, axis=0)

    def _map_from_points(
        self,
        clouds: Tuple[
            List[Tuple[float, float, float]],
            List[Tuple[float, float, float]],
            List[Tuple[float, float, float]],
        ],
    ) -> np.ndarray:
        width, height = self.map_size
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 18)

        def project(pt: Tuple[float, float, float]) -> Tuple[int, int] | None:
            x, _, z = pt
            if z < 0:
                return None
            x = np.clip(x, -self.lateral_range, self.lateral_range)
            z = np.clip(z, 0.0, self.depth_range)
            px = np.interp(
                x,
                [-self.lateral_range, self.lateral_range],
                [20, width - 20],
            )
            py = np.interp(z, [0.0, self.depth_range], [height - 20, 20])
            return int(px), int(py)

        cv2.line(
            canvas,
            (width // 2, height - 20),
            (width // 2, 20),
            (80, 80, 80),
            1,
        )
        cv2.line(
            canvas,
            (20, height - 20),
            (width - 20, height - 20),
            (80, 80, 80),
            1,
        )
        cv2.putText(
            canvas,
            "Depth (m)",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )

        colors = [(0, 170, 255), (255, 120, 0)]
        predicted_color = (255, 0, 255)

        for cam_idx, points in enumerate(clouds[:2]):
            for point in points:
                projected = project(point)
                if projected is None:
                    continue
                cv2.circle(canvas, projected, 4, colors[cam_idx], -1)

        if len(clouds) > 2 and clouds[2]:
            for point in clouds[2]:
                projected = project(point)
                if projected is None:
                    continue
                cv2.circle(canvas, projected, 4, predicted_color, -1)

        return canvas
