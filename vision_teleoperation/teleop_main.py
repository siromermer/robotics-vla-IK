#!/usr/bin/env python3
"""Vision teleoperation: hand video to VX300s MuJoCo replay and MP4 export.

Purpose
    Perception pipeline: MediaPipe hands, YOLO cup, Depth Anything V2,
    map wrist/hand signals to end-effector pose + gripper, solve IK, step the
    sim, compose a 3-panel video (RGB+overlays | depth | robot).

Who calls this
    Run as ``python -m vision_teleoperation.teleop_main`` from the repo root.

Main data shapes
    ``bgr_frames``: list of (H, W, 3) uint8 BGR frames.
    ``depth_maps[i]``: (H, W) float32 in [0, 1], larger = closer (after norm).
    ``positions``: (N, 3) EE target positions in metres (robot base frame).
    ``orientations``: (N, 3, 3) EE target rotation matrices.
    ``gripper_ctrl``: (N,) metres for the parallel gripper actuator.
    ``joint_traj``: (N, 6) solved joint angles in radians.
"""

from __future__ import annotations

import os
import shutil
import sys

if not os.environ.get("MUJOCO_GL"):
    if sys.platform.startswith("linux") and shutil.which("nvidia-smi"):
        # Use EGL on NVIDIA Linux for headless MuJoCo rendering.
        os.environ["MUJOCO_GL"] = "egl"
    else:
        # Use GLFW when a display is available (typical desktop setup).
        os.environ["MUJOCO_GL"] = "glfw"

_nv = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if os.environ.get("MUJOCO_GL") == "egl" and os.path.isfile(_nv):
    # Pin EGL vendor to NVIDIA when present (avoids Mesa vs NVIDIA confusion).
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _nv)

import argparse
from pathlib import Path

import cv2
import imageio.v3 as iio
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

from vision_teleoperation.cup_detector import CupDetector
from vision_teleoperation.depth_estimator import DepthEstimator
from vision_teleoperation.hand_tracker import HandTracker
from vision_teleoperation.vx300s_ik import (
    HOME_QPOS,
    NUM_JOINTS,
    forward_kinematics,
    rot_x,
    rot_y,
    solve_ik_trajectory,
)

WS_X = (0.15, 0.45)
WS_Y = (-0.25, 0.25)
WS_Z = (0.05, 0.50)
# Conservative workspace bounds (metres) to keep IK targets reachable.


def _clamp_to_workspace(positions: np.ndarray) -> np.ndarray:
    """Clip EE positions to axis-aligned workspace boxes.

    Input
        ``positions``: (N, 3) metres, columns x, y, z.
    Output
        Same array, clipped in-place per axis to ``WS_*`` tuples.
    Called by
        ``build_trajectory`` after mapping hand signals to ``positions``.
    """
    positions[:, 0] = np.clip(positions[:, 0], *WS_X)  # reach (x)
    positions[:, 1] = np.clip(positions[:, 1], *WS_Y)
    positions[:, 2] = np.clip(positions[:, 2], *WS_Z)
    return positions


def _hand_pitch_roll(world_pts: np.ndarray) -> tuple[float, float]:
    """Scalar pitch and roll from MediaPipe world landmarks (hand-centred 3D).

    Input
        ``world_pts``: (21, 3) metres, MediaPipe order; uses indices 0,5,9,17.
    Output
        ``(pitch, roll)`` radians for palm attitude relative to MediaPipe axes.
    Called by
        ``build_trajectory`` only.
    """
    wrist = world_pts[0]
    mid_mcp = world_pts[9]
    idx_mcp = world_pts[5]
    pnk_mcp = world_pts[17]

    fwd = mid_mcp - wrist  # palm "forward" direction estimate
    fwd_n = np.linalg.norm(fwd)
    if fwd_n < 1e-7:
        return 0.0, 0.0
    fwd = fwd / fwd_n

    pitch = float(np.arcsin(np.clip(-fwd[1], -1, 1)))  # tilt up/down

    side = pnk_mcp - idx_mcp  # left/right direction across palm
    side = side - np.dot(side, fwd) * fwd
    sn = np.linalg.norm(side)
    if sn < 1e-7:
        return pitch, 0.0
    side = side / sn

    up = np.array([0.0, -1.0, 0.0])  # MediaPipe world "up" reference
    up = up - np.dot(up, fwd) * fwd
    un = np.linalg.norm(up)
    if un < 1e-7:
        return pitch, 0.0
    up = up / un

    cos_r = np.clip(float(np.dot(side, up)), -1, 1)
    sin_r = float(np.dot(np.cross(up, side), fwd))
    roll = float(np.arctan2(sin_r, cos_r))

    return pitch, roll


def build_trajectory(
    hand_frames,
    cup_detections,
    depth_maps: list[np.ndarray],
    fps: float,
):
    """Turn per-frame vision into EE pose targets and gripper commands.

    Inputs
        ``hand_frames``: length N list of ``HandFrame`` from ``HandTracker``.
        ``cup_detections``: length N list of ``CupDetection`` from ``CupDetector``.
        ``depth_maps``: length N list of (H, W) float32 depth maps.
        ``fps``: video FPS, used only for Gaussian smoothing sigma.

    Outputs
        ``positions`` (N, 3), ``orientations`` (N, 3, 3), ``gripper_ctrl`` (N,).

    Mapping idea
        Lateral EE y,z from cup-anchored image offsets (wrist minus cup centre).
        EE x from a blend of apparent hand size and wrist depth (so moving the
        grasped cup still changes x). Orientation from amplified pitch/roll
        deltas vs first detected frame. Gripper uses MediaPipe pinch metric
        with hysteresis while ``locked`` after a firm close.

    Called by
        ``main()`` in this module after hand, cup, and depth steps.
    """
    N = len(hand_frames)
    fk_home = forward_kinematics(HOME_QPOS)  # home EE pose is the mapping reference
    ee_home_pos = fk_home.position.copy()
    ee_home_rot = fk_home.rotation.copy()

    detected_hand = np.array([f.detected for f in hand_frames])
    first_det = int(np.argmax(detected_hand)) if detected_hand.any() else -1
    if first_det < 0:
        # Mapping needs a reference frame; without any hand detection we cannot proceed.
        raise RuntimeError("No hand detected in any frame")

    wrist_uv = np.full((N, 2), np.nan)
    cup_uv = np.full((N, 2), np.nan)
    depth_wrist = np.full(N, np.nan)
    depth_cup = np.full(N, np.nan)
    hand_sizes = np.full(N, np.nan)
    pitches = np.full(N, np.nan)
    rolls = np.full(N, np.nan)
    grippers = np.full(N, np.nan)

    for i in range(N):
        hf = hand_frames[i]
        cd = cup_detections[i]
        dm = depth_maps[i]

        if hf.detected:
            # Only populate these channels when landmarks are available.
            wrist_uv[i] = hf.wrist_uv
            pitches[i], rolls[i] = _hand_pitch_roll(hf.world_landmarks)
            grippers[i] = hf.gripper_opening
            hand_sizes[i] = hf.hand_size_image
            depth_wrist[i] = DepthEstimator.sample_depth_at(dm, hf.wrist_uv[0], hf.wrist_uv[1])

        if cd.detected and cd.center_uv is not None:
            # Cup detections can exist even when the hand is missed.
            cup_uv[i] = cd.center_uv
            if cd.bbox is not None:
                depth_cup[i] = DepthEstimator.sample_depth_bbox(dm, cd.bbox)

    all_series = [
        wrist_uv[:, 0], wrist_uv[:, 1],
        cup_uv[:, 0], cup_uv[:, 1],
        depth_wrist, depth_cup,
        hand_sizes,
        pitches, rolls, grippers,
    ]
    for arr in all_series:
        valid = ~np.isnan(arr)
        if valid.sum() < 2:
            # If there are too few valid samples, fall back to a constant value.
            val = arr[np.where(valid)[0][0]] if valid.any() else 0.5
            arr[:] = val
            continue
        # Fill missing frames by linear interpolation across time (frame index).
        arr[:] = np.interp(np.arange(N), np.where(valid)[0], arr[valid])

    for arr in all_series:
        # Before the first hand detection, hold the first detected value constant.
        arr[:first_det] = arr[first_det]

    sigma = max(fps * 0.10, 1.5)  # smoothing in frames (~100ms)
    for arr in all_series:
        arr[:] = gaussian_filter1d(arr, sigma)

    rel_u = wrist_uv[:, 0] - cup_uv[:, 0]  # cup-anchored wrist motion (robust to camera drift)
    rel_v = wrist_uv[:, 1] - cup_uv[:, 1]

    rel_u0 = rel_u[first_det]
    rel_v0 = rel_v[first_det]

    s0 = hand_sizes[first_det]
    dw0 = depth_wrist[first_det]

    SCALE_Y = 0.30  # metres per normalized u-delta
    SCALE_Z = 0.35  # metres per normalized v-delta
    SCALE_X = 0.35  # metres per depth proxy delta

    positions = np.tile(ee_home_pos, (N, 1))
    for i in range(N):
        du = rel_u[i] - rel_u0
        dv = rel_v[i] - rel_v0

        size_ratio = s0 / max(hand_sizes[i], 0.005)  # apparent-size depth cue
        dd_size = size_ratio - 1.0

        dd_depth = -(depth_wrist[i] - dw0)  # DepthAnything wrist depth cue

        # Blend reach cues: hand size is often stable while grasping; depth adds extra signal.
        dd = 0.6 * dd_size + 0.4 * dd_depth

        positions[i, 0] = ee_home_pos[0] + dd * SCALE_X
        positions[i, 1] = ee_home_pos[1] - du * SCALE_Y
        positions[i, 2] = ee_home_pos[2] - dv * SCALE_Z

    positions = _clamp_to_workspace(positions)  # prevent unreachable IK targets

    pitch0 = pitches[first_det]
    roll0 = rolls[first_det]

    PITCH_GAIN = 1.8  # amplify pitch response
    ROLL_GAIN = 1.5   # amplify roll response
    MAX_PITCH = 0.9
    MAX_ROLL = 0.7

    orientations = np.tile(ee_home_rot, (N, 1, 1))
    for i in range(N):
        dp = np.clip((pitches[i] - pitch0) * PITCH_GAIN, -MAX_PITCH, MAX_PITCH)
        dr = np.clip((rolls[i] - roll0) * ROLL_GAIN, -MAX_ROLL, MAX_ROLL)
        delta_R = rot_y(-dp) @ rot_x(dr)  # incremental wrist orientation relative to home
        orientations[i] = ee_home_rot @ delta_R

    GRIP_CLOSE = 0.25
    GRIP_OPEN = 0.55
    locked = False  # gripper hysteresis state (prevents jitter-open after closing)
    for i in range(N):
        if not locked and grippers[i] < GRIP_CLOSE:
            locked = True
        elif locked and grippers[i] > GRIP_OPEN:
            locked = False
        if locked:
            grippers[i] = min(grippers[i], GRIP_CLOSE * 0.4)

    GRIP_MIN, GRIP_MAX = 0.021, 0.057
    gripper_ctrl = GRIP_MIN + grippers * (GRIP_MAX - GRIP_MIN)

    print(f"[trajectory] {N} frames, first_det={first_det}")
    print(f"[trajectory] hand_size range: [{hand_sizes.min():.4f}, {hand_sizes.max():.4f}] (ref={s0:.4f})")
    print(f"[trajectory] pos X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"[trajectory] pos Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    print(f"[trajectory] pos Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
    print(f"[trajectory] pitch range: [{pitches.min():.3f}, {pitches.max():.3f}] (ref={pitch0:.3f})")
    print(f"[trajectory] roll  range: [{rolls.min():.3f}, {rolls.max():.3f}] (ref={roll0:.3f})")

    return positions, orientations, gripper_ctrl


def simulate_and_render(
    joint_traj: np.ndarray,
    gripper_ctrl: np.ndarray,
    fps: float,
    render_w: int = 480,
    render_h: int = 360,
) -> list[np.ndarray]:
    """Step MuJoCo with solved joints and render RGB frames.

    Inputs
        ``joint_traj``: (N, 6) radians, waist through wrist_rotate.
        ``gripper_ctrl``: (N,) metres, single actuator for parallel gripper.
        ``fps``: used to choose integrator steps per rendered frame.
        ``render_w``, ``render_h``: output image size.

    Output
        List of N BGR uint8 images (H=render_h, W=render_w).

    Called by
        ``main()`` after ``solve_ik_trajectory``.
    """
    scene_xml = Path(__file__).resolve().parent / "assets" / "trossen_vx300s" / "scene.xml"  # right-arm scene
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    renderer = mujoco.Renderer(model, width=render_w, height=render_h)  # offscreen RGB renderer

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 0.9
    cam.azimuth = 150
    cam.elevation = -25
    cam.lookat[:] = [0.15, 0.0, 0.15]

    sim_steps_per_frame = max(int(round(1.0 / (fps * model.opt.timestep))), 1)  # keep video FPS stable
    N = len(joint_traj)
    frames: list[np.ndarray] = []

    for i in range(N):
        data.ctrl[:NUM_JOINTS] = joint_traj[i]  # position actuators for the 6 joints
        data.ctrl[NUM_JOINTS] = gripper_ctrl[i]  # single gripper actuator

        for _ in range(sim_steps_per_frame):
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        if i % 100 == 0:
            print(f"  [sim] rendered {i}/{N}")

    renderer.close()
    print(f"  [sim] done: {len(frames)} frames")
    return frames


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def annotate_hand_frame(
    frame: np.ndarray,
    hand_frame,
    cup_det,
    frame_idx: int,
) -> np.ndarray:
    """Draw hand skeleton, cup box, and debug text on one BGR frame.

    Inputs
        ``frame``: (H, W, 3) BGR source.
        ``hand_frame``: ``HandFrame`` for this index.
        ``cup_det``: ``CupDetection`` for this index.
        ``frame_idx``: integer label in corner.

    Output
        BGR image same size as ``frame``.

    Called by
        ``compose_side_by_side``.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    if cup_det.detected and cup_det.bbox is not None:
        x1, y1, x2, y2 = cup_det.bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cx, cy = int(cup_det.center_uv[0] * w), int(cup_det.center_uv[1] * h)
        cv2.drawMarker(out, (cx, cy), (0, 165, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.putText(out, f"CUP {cup_det.confidence:.0%}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)

    if hand_frame.detected:
        pts = hand_frame.image_landmarks
        px = (pts[:, 0] * w).astype(int)
        py = (pts[:, 1] * h).astype(int)
        for s, e in HAND_CONNECTIONS:
            if s < len(pts) and e < len(pts):
                cv2.line(out, (px[s], py[s]), (px[e], py[e]), (255, 180, 0), 2)
        for j in range(len(pts)):
            color = (0, 255, 0) if j == 0 else (0, 200, 255)
            cv2.circle(out, (px[j], py[j]), 3, color, -1)
        cv2.circle(out, (px[0], py[0]), 7, (0, 255, 0), 2)

        txt = f"grip={hand_frame.gripper_opening:.2f}"
        cv2.putText(out, txt, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(out, "NO HAND", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.putText(out, f"#{frame_idx}", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def compose_side_by_side(
    hand_bgr_frames: list[np.ndarray],
    sim_bgr_frames: list[np.ndarray],
    depth_maps: list[np.ndarray],
    hand_data,
    cup_data,
    target_h: int = 360,
) -> list[np.ndarray]:
    """Horizontally stack annotated RGB, depth colormap, and sim frames.

    Inputs
        ``hand_bgr_frames``, ``sim_bgr_frames``, ``depth_maps``: length N lists;
        depth maps are (H, W) float in [0, 1].
        ``hand_data``, ``cup_data``: per-frame tracker outputs, same length.
        ``target_h``: resize all panels to this height; width scales to keep AR.

    Output
        List of N BGR images, width is sum of three panel widths at ``target_h``.

    Called by
        ``main()`` for the final MP4.
    """
    N = min(len(hand_bgr_frames), len(sim_bgr_frames), len(depth_maps))
    composed: list[np.ndarray] = []

    for i in range(N):
        left = annotate_hand_frame(hand_bgr_frames[i], hand_data[i], cup_data[i], i)
        dm = depth_maps[i]
        dm_u8 = np.clip(dm * 255.0, 0, 255).astype(np.uint8)
        depth_bgr = cv2.applyColorMap(dm_u8, cv2.COLORMAP_TURBO)

        right = sim_bgr_frames[i]

        lh, lw = left.shape[:2]
        dh, dw = depth_bgr.shape[:2]
        rh, rw = right.shape[:2]
        left = cv2.resize(left, (int(lw * target_h / lh), target_h))
        depth_bgr = cv2.resize(depth_bgr, (int(dw * target_h / dh), target_h))
        right = cv2.resize(right, (int(rw * target_h / rh), target_h))

        cv2.putText(left, "Hand Input + YOLO", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(depth_bgr, "DepthAnythingV2 (relative)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(right, "VX300s Sim", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        composed.append(np.hstack([left, depth_bgr, right]))

    return composed


def main() -> None:
    """CLI: run the full pipeline and write an MP4 under ``vision_teleoperation/results/``."""
    parser = argparse.ArgumentParser(
        description="Vision-based teleoperation: hand video -> VX300s robot sim"
    )
    parser.add_argument("--video", required=True, help="Input hand video path")
    parser.add_argument(
        "--output", default="vision_teleoperation/results/teleop_hand_yolo_depth_robot.mp4",
        help="Output MP4 path (3-panel: hand+YOLO | depth | sim)",
    )
    parser.add_argument("--fps", type=float, default=0,
                        help="Override output FPS (0 = use video FPS)")
    parser.add_argument("--render-width", type=int, default=480)
    parser.add_argument("--render-height", type=int, default=360)
    args = parser.parse_args()

    print(f"[main] MUJOCO_GL = {os.environ.get('MUJOCO_GL', '(unset)')}")

    print("\n[1/7] Hand tracking (MediaPipe)")
    tracker = HandTracker()
    hand_data, bgr_frames, vid_fps = tracker.process_video(args.video)
    out_fps = args.fps if args.fps > 0 else vid_fps

    print("\n[2/7] Cup detection (YOLOv8n)")
    cup_det = CupDetector()
    cup_data = cup_det.detect_video(bgr_frames)

    print("\n[3/7] Depth (DepthAnythingV2 Small)")
    depth_est = DepthEstimator()
    depth_maps = depth_est.estimate_video(bgr_frames)

    print("\n[4/7] Trajectory mapping (hand -> EE targets)")
    positions, orientations, gripper_ctrl = build_trajectory(
        hand_data, cup_data, depth_maps, vid_fps,
    )

    print("\n[5/7] Inverse kinematics (DLS, from scratch)")
    joint_traj = solve_ik_trajectory(
        positions, orientations,
        q_init=HOME_QPOS.copy(),
        max_iter=200,
        damping=0.05,
        orient_weight=0.5,
    )

    print("\n[6/7] MuJoCo simulation + render")
    sim_frames = simulate_and_render(
        joint_traj, gripper_ctrl, out_fps,
        render_w=args.render_width, render_h=args.render_height,
    )

    print("\n[7/7] Compose MP4")
    composed = compose_side_by_side(
        bgr_frames, sim_frames, depth_maps, hand_data, cup_data,
        target_h=args.render_height,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rgb_stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in composed])
    iio.imwrite(str(out_path), rgb_stack, fps=out_fps)
    print(f"\n[done] Saved {len(composed)} frames -> {out_path}")
    print(f"[done] Resolution: {composed[0].shape[1]}x{composed[0].shape[0]} @ {out_fps:.1f} fps")


if __name__ == "__main__":
    main()
