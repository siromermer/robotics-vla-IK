#!/usr/bin/env python3
"""YOLOv8n cup detection on a list of BGR video frames (GPU if available).

Purpose
    Provide a stable image-space anchor (cup bbox and centre) so wrist motion
    can be expressed relative to the object, reducing apparent drift when the
    camera moves slightly.

Who calls this
    ``CupDetector.detect_video`` from ``task2.teleop_main.main``.

Weights
    Default ``task2/assets/yolov8n.pt`` (Ultralytics). Class id 41 is COCO ``cup``.

Data shapes
    ``detect_video`` input: list of (H, W, 3) uint8 BGR, length N.
    Output: list of N ``CupDetection`` objects.
    ``CupDetection.bbox``: (4,) float pixels x1,y1,x2,y2 or None.
    ``CupDetection.center_uv``: (2,) normalized [0,1] or None.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "assets" / "yolov8n.pt"
# Local weights are stored under task2/assets so the repo runs offline.

CUP_CLASS_ID = 41


@dataclass
class CupDetection:
    """Per-frame best cup detection (highest confidence), if any."""

    detected: bool = False
    bbox: np.ndarray | None = None
    center_uv: np.ndarray | None = None
    confidence: float = 0.0
    bbox_size: float = 0.0


class CupDetector:
    """Batch-inference wrapper around Ultralytics YOLO."""

    def __init__(self, weights: str | Path | None = None, conf: float = 0.25):
        w = Path(weights) if weights else DEFAULT_WEIGHTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(w))
        self.model.to(device)
        self.conf = conf
        # We restrict to COCO "cup" class to keep detections stable and fast.
        print(f"[cup_detector] YOLOv8n on {device}, conf={conf}")

    def detect_video(self, bgr_frames: list[np.ndarray]) -> list[CupDetection]:
        """Run cup-only detection on all frames in batches of 16.

        Returns
            List aligned with ``bgr_frames``; each element is ``CupDetection``.

        Called by
            ``task2.teleop_main.main``.
        """
        N = len(bgr_frames)
        results_list: list[CupDetection] = []

        batch_size = 16  # small batches reduce GPU memory spikes on long videos
        for start in range(0, N, batch_size):
            batch = bgr_frames[start : start + batch_size]
            preds = self.model.predict(
                batch,
                conf=self.conf,
                classes=[CUP_CLASS_ID],
                verbose=False,
                imgsz=320,
            )

            for pred in preds:
                det = CupDetection()
                boxes = pred.boxes
                if boxes is not None and len(boxes) > 0:
                    confs = boxes.conf.cpu().numpy()
                    best = int(np.argmax(confs))
                    xyxy = boxes.xyxy[best].cpu().numpy()

                    h, w = pred.orig_shape
                    # Convert bbox center to normalized [0,1] so it aligns with hand wrist_uv.
                    cx = (xyxy[0] + xyxy[2]) / 2.0
                    cy = (xyxy[1] + xyxy[3]) / 2.0
                    bw = xyxy[2] - xyxy[0]
                    bh = xyxy[3] - xyxy[1]

                    det.detected = True
                    det.bbox = xyxy
                    det.center_uv = np.array([cx / w, cy / h])
                    det.confidence = float(confs[best])
                    det.bbox_size = float(max(bw, bh) / max(w, h))

                results_list.append(det)

            if (start + batch_size) % 100 < batch_size:
                print(f"  [cup_detector] {min(start + batch_size, N)}/{N}")

        n_det = sum(1 for d in results_list if d.detected)
        print(f"  [cup_detector] done: {n_det}/{N} frames with cup")
        return results_list


if __name__ == "__main__":
    import sys
    import cv2

    if len(sys.argv) < 2:
        print("usage: python cup_detector.py <video>")
        raise SystemExit(1)

    cap = cv2.VideoCapture(sys.argv[1])
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    det = CupDetector()
    results = det.detect_video(frames)
    for i, r in enumerate(results):
        if r.detected:
            print(f"  frame {i:4d}: cup at ({r.center_uv[0]:.3f}, {r.center_uv[1]:.3f}) "
                  f"conf={r.confidence:.2f} size={r.bbox_size:.3f}")
