#!/usr/bin/env python3
"""Depth Anything V2 Small (Hugging Face) for monocular relative depth maps.

Purpose
    Per-frame dense depth-like maps (ordinal, not metric) for sampling depth at
    the wrist and inside the cup bounding box. Used together with hand size
    in ``task2.teleop_main.build_trajectory``.

Who calls this
    ``DepthEstimator.estimate_video`` from ``task2.teleop_main.main``.

Inputs and outputs
    ``estimate_video(bgr_frames)``:
        Input: list of (H, W, 3) uint8 BGR, length N.
        Output: list of N arrays (H, W) float32, values per frame linearly
        scaled to [0, 1] where larger means closer in that frame.

    ``sample_depth_at(depth_map, u, v)``: ``u,v`` normalized [0,1]; returns scalar.
    ``sample_depth_bbox(depth_map, bbox)``: ``bbox`` (4,) pixels x1,y1,x2,y2;
        returns mean depth in the rectangle.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    """Wraps HF ``depth-estimation`` pipeline on CUDA fp16 when available."""

    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        # HF pipeline handles model download/caching automatically (first run can take time).
        self.pipe = pipeline(
            "depth-estimation",
            model=MODEL_ID,
            device=device,
            dtype=torch.float16,
        )
        tag = "cuda" if device == 0 else "cpu"
        print(f"[depth] DepthAnythingV2-Small on {tag}")

    def estimate_video(
        self,
        bgr_frames: list[np.ndarray],
        target_size: int = 384,
    ) -> list[np.ndarray]:
        """Infer depth for each frame; resize internally then back to full res.

        ``target_size`` is the max side length fed to the model for speed.

        Called by
            ``task2.teleop_main.main``.
        """
        N = len(bgr_frames)
        depth_maps: list[np.ndarray] = []

        for i, bgr in enumerate(bgr_frames):
            h, w = bgr.shape[:2]
            scale = target_size / max(h, w)
            # Run the model on a downscaled frame for speed, then upsample back to full res.
            small = cv2.resize(bgr, (int(w * scale), int(h * scale)))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            result = self.pipe(pil_img)
            depth_pil = result["depth"]
            depth_np = np.array(depth_pil, dtype=np.float32)

            d_min, d_max = depth_np.min(), depth_np.max()
            if d_max - d_min > 1e-6:
                # Per-frame normalization to [0,1] makes this an ordinal depth cue, not metric.
                depth_np = (depth_np - d_min) / (d_max - d_min)
            else:
                depth_np[:] = 0.5

            depth_full = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)
            depth_maps.append(depth_full)

            if i % 100 == 0:
                print(f"  [depth] {i}/{N}")

        print(f"  [depth] done: {N} frames")
        return depth_maps

    @staticmethod
    def sample_depth_at(
        depth_map: np.ndarray,
        u: float,
        v: float,
        patch_radius: int = 3,
    ) -> float:
        """Mean depth in a small square around pixel for ``(u,v)`` normalized."""
        h, w = depth_map.shape[:2]
        px = int(np.clip(u * w, 0, w - 1))
        py = int(np.clip(v * h, 0, h - 1))
        r = patch_radius
        y0 = max(0, py - r)
        y1 = min(h, py + r + 1)
        x0 = max(0, px - r)
        x1 = min(w, px + r + 1)
        return float(np.mean(depth_map[y0:y1, x0:x1]))

    @staticmethod
    def sample_depth_bbox(
        depth_map: np.ndarray,
        bbox: np.ndarray,
    ) -> float:
        """Mean depth over axis-aligned bbox in pixel coordinates."""
        h, w = depth_map.shape[:2]
        x1 = int(np.clip(bbox[0], 0, w - 1))
        y1 = int(np.clip(bbox[1], 0, h - 1))
        x2 = int(np.clip(bbox[2], 0, w))
        y2 = int(np.clip(bbox[3], 0, h))
        if x2 <= x1 or y2 <= y1:
            return 0.5
        return float(np.mean(depth_map[y1:y2, x1:x2]))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python depth_estimator.py <video>")
        raise SystemExit(1)

    cap = cv2.VideoCapture(sys.argv[1])
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    est = DepthEstimator()
    maps = est.estimate_video(frames[:20])
    for i, dm in enumerate(maps):
        print(f"  frame {i}: depth shape={dm.shape}, "
              f"range=[{dm.min():.3f}, {dm.max():.3f}]")
