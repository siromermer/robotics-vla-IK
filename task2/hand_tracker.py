#!/usr/bin/env python3
"""Google MediaPipe Hand Landmarker on a video file (Task 2 vision input).

Purpose
    Produce per-frame 2D and 3D hand landmarks plus simple derived signals
    (wrist image position, grip opening, apparent hand size) for teleoperation.

Who calls this
    ``HandTracker.process_video`` is invoked from ``task2.teleop_main.main``.

Model file
    ``hand_landmarker.task`` next to this module (downloaded automatically
    from Google Storage on first run if missing).

Main output shapes
    Each ``HandFrame`` holds:
    ``wrist_uv`` (2,) normalized [0,1] image coords of landmark 0;
    ``world_landmarks`` (21, 3) metres in MediaPipe hand-centred frame;
    ``image_landmarks`` (21, 2) normalized image coords;
    ``orientation`` (3, 3) palm frame (optional, not used by current mapper);
    ``gripper_opening`` scalar in [0, 1] from thumb-index distance;
    ``hand_size_image`` scalar, norm of wrist to middle MCP in image space.

    ``process_video`` also returns ``bgr_frames``: list of (H,W,3) uint8 frames
    and ``fps``: float.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"
# The .task file is treated as a runtime asset (kept next to this module).

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


@dataclass
class HandFrame:
    """One frame of hand tracking; see module docstring for field shapes."""

    detected: bool = False
    wrist_uv: np.ndarray = field(default_factory=lambda: np.zeros(2))
    world_landmarks: np.ndarray = field(default_factory=lambda: np.zeros((21, 3)))
    image_landmarks: np.ndarray = field(default_factory=lambda: np.zeros((21, 2)))
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3))
    gripper_opening: float = 0.5
    hand_size_image: float = 0.0


class HandTracker:
    """Loads the task model and runs VIDEO mode landmarker over all frames."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._ensure_model()

    def _ensure_model(self) -> None:
        """Download ``hand_landmarker.task`` if path does not exist."""
        if self.model_path.exists():
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        # First run convenience: download the model automatically if missing.
        print(f"[hand_tracker] downloading model -> {self.model_path}")
        urllib.request.urlretrieve(MODEL_URL, self.model_path)

    def process_video(
        self, video_path: str | Path
    ) -> tuple[list[HandFrame], list[np.ndarray], float]:
        """Read ``video_path`` and run the landmarker on every frame.

        Returns
            ``hand_frames``: list of length N (one ``HandFrame`` per frame).
            ``bgr_frames``: list of N arrays (H_i, W_i, 3) uint8 BGR.
            ``fps``: frames per second from container (fallback 25).

        Called by
            ``task2.teleop_main.main``.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.2,
            min_hand_presence_confidence=0.2,
            min_tracking_confidence=0.2,
        )
        # Lower thresholds make detection more permissive on noisy FPP videos.

        hand_frames: list[HandFrame] = []
        bgr_frames: list[np.ndarray] = []
        idx = 0

        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                bgr_frames.append(frame)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = int(idx * 1000.0 / fps)
                # VIDEO mode uses monotonically increasing timestamps (ms).
                result = landmarker.detect_for_video(mp_img, ts)

                hf = HandFrame()
                if result.hand_landmarks and result.hand_world_landmarks:
                    img_lms = result.hand_landmarks[0]
                    wld_lms = result.hand_world_landmarks[0]

                    hf.detected = True
                    # Landmarks are normalized to [0,1] in image space; world landmarks are hand-centred.
                    hf.wrist_uv = np.array([img_lms[0].x, img_lms[0].y])
                    hf.world_landmarks = np.array(
                        [[lm.x, lm.y, lm.z] for lm in wld_lms]
                    )
                    hf.image_landmarks = np.array(
                        [[lm.x, lm.y] for lm in img_lms]
                    )
                    hf.orientation = self._orientation(hf.world_landmarks)
                    hf.gripper_opening = self._gripper(hf.world_landmarks)

                    wrist_img = np.array([img_lms[0].x, img_lms[0].y])
                    mid_img = np.array([img_lms[9].x, img_lms[9].y])
                    # Apparent hand size provides a depth proxy (bigger in image ≈ closer).
                    hf.hand_size_image = float(np.linalg.norm(mid_img - wrist_img))

                hand_frames.append(hf)
                idx += 1

                if idx % 50 == 0:
                    print(f"  [hand_tracker] frame {idx}/{total}")

        cap.release()
        n_det = sum(1 for f in hand_frames if f.detected)
        print(
            f"  [hand_tracker] done: {len(hand_frames)} frames, "
            f"{n_det} detections"
        )
        return hand_frames, bgr_frames, fps

    @staticmethod
    def _orientation(wpts: np.ndarray) -> np.ndarray:
        """Build 3x3 palm rotation from world landmarks (see module docstring).

        Input ``wpts``: (21, 3). Output: (3, 3) columns forward, side, normal.
        Called only from ``process_video``.
        """
        wrist = wpts[0]
        mid_mcp = wpts[9]
        idx_mcp = wpts[5]
        pnk_mcp = wpts[17]

        fwd = mid_mcp - wrist
        n = np.linalg.norm(fwd)
        if n < 1e-7:
            return np.eye(3)
        fwd /= n

        side_raw = pnk_mcp - idx_mcp
        normal = np.cross(fwd, side_raw)
        n = np.linalg.norm(normal)
        if n < 1e-7:
            return np.eye(3)
        normal /= n

        side = np.cross(normal, fwd)
        side /= np.linalg.norm(side)

        return np.column_stack([fwd, side, normal])

    @staticmethod
    def _gripper(wpts: np.ndarray) -> float:
        """Map thumb tip (4) to index tip (8) distance to [0, 1].

        Input ``wpts``: (21, 3) metres. Output: scalar grip openness.
        """
        d = float(np.linalg.norm(wpts[4] - wpts[8]))
        return float(np.clip((d - 0.02) / 0.08, 0.0, 1.0))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python hand_tracker.py <video>")
        raise SystemExit(1)

    tracker = HandTracker()
    frames, _, fps = tracker.process_video(sys.argv[1])
    for i, f in enumerate(frames):
        if f.detected:
            print(
                f"frame {i:4d}: wrist_uv=({f.wrist_uv[0]:.3f}, "
                f"{f.wrist_uv[1]:.3f})  gripper={f.gripper_opening:.2f}  "
                f"hand_size={f.hand_size_image:.4f}"
            )
