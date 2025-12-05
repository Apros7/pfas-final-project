from typing import Dict, List, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from tracking.track_state import TrackState


class KalmanTracker:
    def __init__(
        self,
        max_idle: int = 15,
        min_hits: int = 3,
        match_threshold: float = 4.0,
        dt: float = 1.0,
    ):
        self.max_idle = max_idle
        self.min_hits = min_hits
        self.match_threshold = match_threshold
        self.dt = dt
        self.tracks: Dict[int, Tuple[KalmanFilter, TrackState]] = {}
        self._next_id = 0
        self._frame_idx = 0

    def _create_kalman_filter(
        self,
        x: float,
        y: float,
        z: float,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        is_established: bool = False,
    ) -> KalmanFilter:
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.x = np.array([x, y, z, vx, vy, vz], dtype=np.float32)

        dt = self.dt
        kf.F = np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        if is_established:
            q = 0.05
        else:
            q = 0.1
        kf.Q = np.eye(6, dtype=np.float32) * q
        kf.Q[3:, 3:] *= 0.05 if is_established else 0.1

        r = 0.3 if is_established else 0.5
        kf.R = np.eye(3, dtype=np.float32) * r

        if is_established:
            kf.P = np.eye(6, dtype=np.float32) * 2.0
            kf.P[3:, 3:] *= 1.0
        else:
            kf.P = np.eye(6, dtype=np.float32) * 10.0
            kf.P[3:, 3:] *= 5.0

        return kf

    def _compute_velocity_from_history(
        self, position_history: list, dt: float = 1.0
    ) -> tuple:
        if not position_history or len(position_history) < 2:
            return (0.0, 0.0, 0.0)

        history = (
            position_history[-10:] if len(position_history) > 10 else position_history
        )

        if len(history) < 2:
            return (0.0, 0.0, 0.0)

        positions = np.array([(p[0], p[1], p[2]) for p in history])
        frames = np.array([p[3] for p in history])

        if len(history) >= 2:
            velocities = []
            for i in range(1, len(history)):
                dt_frame = (frames[i] - frames[i - 1]) * dt
                if dt_frame > 0:
                    vx = (positions[i, 0] - positions[i - 1, 0]) / dt_frame
                    vy = (positions[i, 1] - positions[i - 1, 1]) / dt_frame
                    vz = (positions[i, 2] - positions[i - 1, 2]) / dt_frame
                    velocities.append((vx, vy, vz))

            if velocities:
                velocities = np.array(velocities)
                vx = float(np.median(velocities[:, 0]))
                vy = float(np.median(velocities[:, 1]))
                vz = float(np.median(velocities[:, 2]))
                return (vx, vy, vz)

        return (0.0, 0.0, 0.0)

    def _mahalanobis_distance(
        self, kf: KalmanFilter, measurement: np.ndarray
    ) -> float:
        y = measurement - kf.H @ kf.x
        S = kf.H @ kf.P @ kf.H.T + kf.R
        try:
            d = np.sqrt(y.T @ np.linalg.inv(S) @ y)
            return float(d)
        except np.linalg.LinAlgError:
            return float(np.linalg.norm(y))

    def predict(self) -> None:
        self._frame_idx += 1
        for kf, state in self.tracks.values():
            kf.predict()
            state.age += 1
            if state.last_seen < self._frame_idx - 1:
                state.frames_since_last_detection += 1
            else:
                state.frames_since_last_detection = 0

    def update(
        self, measurements: List[Dict[str, float]]
    ) -> Tuple[List[TrackState], List[Dict[str, float]]]:
        if not measurements:
            return [
                state
                for _, state in self.tracks.values()
                if state.age - state.last_seen < self.max_idle
            ], []

        tracks_by_class: Dict[int, List[Tuple[int, KalmanFilter, TrackState]]] = {}
        for track_id, (kf, state) in self.tracks.items():
            if state.age - state.last_seen >= self.max_idle:
                continue
            cls_id = state.cls_id
            tracks_by_class.setdefault(cls_id, []).append((track_id, kf, state))

        matched_measurements: set[int] = set()
        matched_tracks: set[int] = set()

        for cls_id, class_tracks in tracks_by_class.items():
            class_measurements = [
                (idx, m)
                for idx, m in enumerate(measurements)
                if m["class"] == cls_id and idx not in matched_measurements
            ]

            if not class_tracks or not class_measurements:
                continue

            n_tracks = len(class_tracks)
            n_meas = len(class_measurements)
            cost_matrix = np.full((n_tracks, n_meas), np.inf)

            for i, (track_id, kf, state) in enumerate(class_tracks):
                for j, (meas_idx, measurement) in enumerate(class_measurements):
                    meas_vec = np.array(
                        [measurement["x"], measurement["y"], measurement["z"]]
                    )
                    dist = self._mahalanobis_distance(kf, meas_vec)
                    threshold = self.match_threshold
                    if state.frames_since_last_detection > 0:
                        occlusion_factor = min(
                            1.0 + (state.frames_since_last_detection / 10.0), 3.0
                        )
                        threshold = self.match_threshold * occlusion_factor
                    if dist <= threshold:
                        cost_matrix[i, j] = dist

            if n_tracks > 0 and n_meas > 0:
                if np.any(np.isfinite(cost_matrix)):
                    try:
                        row_indices, col_indices = linear_sum_assignment(cost_matrix)

                        for i, j in zip(row_indices, col_indices):
                            if cost_matrix[i, j] < np.inf:
                                track_id, kf, state = class_tracks[i]
                                meas_idx, measurement = class_measurements[j]
                                meas_vec = np.array(
                                    [
                                        measurement["x"],
                                        measurement["y"],
                                        measurement["z"],
                                    ]
                                )

                                if state.position_history is None:
                                    state.position_history = []

                                state.position_history.append(
                                    (
                                        measurement["x"],
                                        measurement["y"],
                                        measurement["z"],
                                        self._frame_idx,
                                    )
                                )

                                if len(state.position_history) > 20:
                                    state.position_history = state.position_history[-20:]

                                is_established = state.stereo_detections > 15
                                if is_established and len(state.position_history) >= 3:
                                    vx_hist, vy_hist, vz_hist = (
                                        self._compute_velocity_from_history(
                                            state.position_history, self.dt
                                        )
                                    )
                                    kf.update(meas_vec)
                                    vx_kf = float(kf.x[3])
                                    vy_kf = float(kf.x[4])
                                    vz_kf = float(kf.x[5])

                                    blend_factor = 0.7
                                    vx = vx_hist * blend_factor + vx_kf * (
                                        1 - blend_factor
                                    )
                                    vy = vy_hist * blend_factor + vy_kf * (
                                        1 - blend_factor
                                    )
                                    vz = vz_hist * blend_factor + vz_kf * (
                                        1 - blend_factor
                                    )

                                    kf.x[3] = vx
                                    kf.x[4] = vy
                                    kf.x[5] = vz

                                    if (
                                        state.stereo_detections > 20
                                        and state.hits % 10 == 0
                                    ):
                                        new_kf = self._create_kalman_filter(
                                            float(kf.x[0]),
                                            float(kf.x[1]),
                                            float(kf.x[2]),
                                            vx,
                                            vy,
                                            vz,
                                            is_established=True,
                                        )
                                        new_kf.P = kf.P.copy()
                                        new_kf.P *= 0.5
                                        kf = new_kf
                                else:
                                    kf.update(meas_vec)

                                state.x = float(kf.x[0])
                                state.y = float(kf.x[1])
                                state.z = float(kf.x[2])
                                state.vx = float(kf.x[3])
                                state.vy = float(kf.x[4])
                                state.vz = float(kf.x[5])
                                state.last_seen = self._frame_idx
                                state.hits += 1
                                state.frames_since_last_detection = 0

                                is_stereo = measurement.get("stereo", True)
                                if is_stereo:
                                    state.stereo_detections += 1
                                    state.single_camera_frames = 0
                                    state.last_detection_type = "stereo"
                                else:
                                    if state.last_detection_type == "single_camera":
                                        state.single_camera_frames += 1
                                    else:
                                        state.single_camera_frames = 1
                                    state.last_detection_type = "single_camera"

                                if "bbox" in measurement:
                                    state.last_bbox = np.array(
                                        measurement["bbox"], dtype=np.float32
                                    )

                                self.tracks[track_id] = (kf, state)

                                matched_tracks.add(track_id)
                                matched_measurements.add(meas_idx)
                    except ValueError:
                        pass

        for idx, measurement in enumerate(measurements):
            if idx not in matched_measurements:
                kf = self._create_kalman_filter(
                    measurement["x"], measurement["y"], measurement["z"]
                )
                is_stereo = measurement.get("stereo", True)
                state = TrackState(
                    track_id=self._next_id,
                    cls_id=measurement["class"],
                    x=measurement["x"],
                    y=measurement["y"],
                    z=measurement["z"],
                    vx=0.0,
                    vy=0.0,
                    vz=0.0,
                    last_seen=self._frame_idx,
                    hits=1,
                    age=0,
                    stereo_detections=1 if is_stereo else 0,
                    single_camera_frames=0 if is_stereo else 1,
                    last_detection_type="stereo" if is_stereo else "single_camera",
                    position_history=[
                        (
                            measurement["x"],
                            measurement["y"],
                            measurement["z"],
                            self._frame_idx,
                        )
                    ],
                )
                self.tracks[self._next_id] = (kf, state)
                self._next_id += 1

        tracks_to_remove = [
            track_id
            for track_id, (_, state) in self.tracks.items()
            if state.age - state.last_seen >= self.max_idle
        ]
        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        active_tracks = []
        for _, state in self.tracks.values():
            if state.hits < self.min_hits:
                continue
            if state.age - state.last_seen >= self.max_idle:
                continue

            if (
                state.stereo_detections > 5
                and state.last_detection_type == "single_camera"
                and state.single_camera_frames > 3
            ):
                continue

            active_tracks.append(state)

        return active_tracks, [
            measurements[idx]
            for idx in matched_measurements
            if idx < len(measurements)
        ]

