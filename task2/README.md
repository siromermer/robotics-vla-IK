## Task 2 (hand video → VX300s right arm) — explanation, conda + run

### What is Task 2?

Task 2 is a **vision-based teleoperation pipeline** that reads a hand-held-camera
video, extracts human hand motion, and replays that motion on a simulated
[Trossen VX300s](https://www.trossenrobotics.com/viperx-300-robot-arm.aspx)
6-DOF robot arm inside [MuJoCo](https://mujoco.org/).  The entire pipeline runs
without any pre-recorded robot demonstrations: it maps raw RGB video to joint
angles entirely through perception, geometry, and a **from-scratch inverse
kinematics (IK)** solver.

---

### Pipeline overview (7 steps)

```
Input video (RGB)
     │
     ▼
[1] Hand tracking (MediaPipe)   → per-frame landmarks, wrist UV, grip signal
     │
     ▼
[2] Cup detection (YOLOv8n)     → per-frame cup bbox + normalised centre UV
     │
     ▼
[3] Depth estimation (Depth-Anything-V2-Small)  → per-frame (H×W) relative depth map
     │
     ▼
[4] Trajectory mapping          → (N,3) EE positions, (N,3,3) orientations, (N,) gripper
     │
     ▼
[5] Inverse kinematics (DLS)    → (N,6) joint angles
     │
     ▼
[6] MuJoCo simulation + render  → (N,) robot RGB frames
     │
     ▼
[7] Compose 3-panel MP4
     │
     ▼
Output: hand+YOLO | depth colourmap | robot sim
```

---

### Component descriptions

#### Step 1 — Hand tracking (`hand_tracker.py`)

Uses **Google MediaPipe Hand Landmarker** in *VIDEO* mode (monotonic
timestamps) to locate 21 hand landmarks per frame.

Key outputs stored in each `HandFrame`:

| Field | Shape | Description |
|---|---|---|
| `wrist_uv` | (2,) | Normalised [0,1] image coords of landmark 0 (wrist) |
| `world_landmarks` | (21, 3) m | Hand-centred 3-D landmarks in metres |
| `image_landmarks` | (21, 2) | Normalised image coords of all 21 landmarks |
| `gripper_opening` | scalar | Thumb-tip ↔ index-tip distance mapped to [0,1] |
| `hand_size_image` | scalar | Wrist → middle-MCP distance in image space (depth proxy) |

Detection thresholds are kept low (0.2) to remain robust on first-person
hand videos where the hand is partially occluded.

#### Step 2 — Cup detection (`cup_detector.py`)

Runs **YOLOv8n** (COCO class 41 = *cup*) on every frame (batch size 16).
Detections provide a stable image-space anchor so that wrist motion is
expressed *relative to the cup* rather than in absolute pixel space.  This
makes the mapping robust to minor camera shake or drift.

#### Step 3 — Depth estimation (`depth_estimator.py`)

Runs **Depth-Anything-V2-Small** (Hugging Face) to produce a per-frame
`(H, W) float32` relative depth map normalised to [0, 1] (larger value =
closer).  Two scalars are sampled per frame: depth at the wrist (`sample_depth_at`)
and mean depth inside the cup bounding box (`sample_depth_bbox`).  These
ordinal depth cues supplement the apparent hand-size depth cue.

#### Step 4 — Trajectory mapping (`teleop_main.build_trajectory`)

Converts all per-frame vision signals into **end-effector (EE) pose targets**
relative to the robot's home position:

| Signal | Robot axis | Scale |
|---|---|---|
| Cup-anchored wrist Δu (horizontal) | EE y (left/right) | 0.30 m/unit |
| Cup-anchored wrist Δv (vertical) | EE z (up/down) | 0.35 m/unit |
| Blend of apparent-size + depth Δ | EE x (reach) | 0.35 m/unit |

Orientation deltas (pitch and roll) are computed from the palm plane using
landmarks 0, 5, 9, 17 and applied as incremental `Ry(-Δpitch) @ Rx(Δroll)`
rotations relative to the home EE rotation.

A hysteresis state machine converts the `gripper_opening` scalar into
a MuJoCo parallel-gripper actuator command (metres) with open/close thresholds.

All signals are gap-filled by linear interpolation and smoothed with a
Gaussian filter (σ ≈ 100 ms) before mapping.  EE targets are clamped to a
conservative workspace box to keep every IK target reachable:

```
x ∈ [0.15, 0.45] m   (reach)
y ∈ [−0.25, 0.25] m  (lateral)
z ∈ [0.05, 0.50] m   (height)
```

#### Step 5 — From-scratch IK (`vx300s_ik.py`)

The IK solver is implemented **entirely from scratch** (no robotics library).
It mirrors the MuJoCo Menagerie `vx300s.xml` kinematic tree exactly.

**Joint vector** `q ∈ ℝ⁶` (radians):
`waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate`

**Forward kinematics (FK)**

Iterates through the 6 joints, accumulating a 4×4 homogeneous transform:

```
T ← T · Translate(LINK_OFFSET_i) · Rotate(axis_i, q_i)
```

followed by a final pinch-site offset `EE_OFFSET = [0.1, 0, 0]`.  Each
joint origin and world-frame axis are recorded for Jacobian construction.

**Geometric Jacobian** `J ∈ ℝ^{6×6}`

For each revolute joint *i* with world axis `z_i` and origin `p_i`:

```
J[:3, i] = z_i × (p_ee − p_i)   (linear velocity contribution, p_ee = EE position in world frame)
J[3:, i] = z_i                   (angular velocity contribution)
```

**Damped least-squares (DLS) IK**

The position + orientation error is stacked into a 6-vector:

```
e = [e_pos;  w_orient · e_ori]
```

where `e_ori` is the rotation-vector form of `R_err = R_target · R_current^T`.

The update rule (one Newton step):

```
dq = J^T · (J · J^T + λ² · I)⁻¹ · e
q  ← clip(q + dq, joint_limits)
```

`λ = 0.05` (damping) prevents blow-up near singularities.  The default
orientation weight is `w_orient = 0.5`.

Each frame warm-starts from the previous frame's solution, so the solver
typically converges in a small number of iterations for smooth trajectories.

#### Step 6 — MuJoCo simulation + render (`teleop_main.simulate_and_render`)

Loads `task2/assets/trossen_vx300s/scene.xml`, sets the 6 joint position
actuators plus 1 gripper actuator per frame, steps the physics forward, and
renders an off-screen RGB image with `mujoco.Renderer` at 480×360.

A free camera is fixed at azimuth 150°, elevation −25°, distance 0.9 m for
a clear overhead-diagonal view of the arm.

#### Step 7 — 3-panel video composition (`teleop_main.compose_side_by_side`)

Three panels are resized to the same height and horizontally stacked:

| Panel | Content |
|---|---|
| Left | Original hand video with MediaPipe skeleton overlay + YOLO cup box |
| Centre | Depth map colourised with `COLORMAP_TURBO` |
| Right | MuJoCo robot simulation render |

---

### Conda environment (name: `lerobot-task2`)

From inside `submission/task2/`:

```bash
conda create -n lerobot-task2 python=3.10 -y
conda activate lerobot-task2
pip install -r requirements.txt
```

### Run (activate env → python)

From inside `submission/` (with env activated):

```bash
cd task2
export MUJOCO_GL=egl
export PYTHONPATH="$(pwd)/.."

python -m task2.teleop_main \
  --video task2/test_video.mp4 \
  --output task2/results/teleop_hand_yolo_depth_robot.mp4
```

### Output

- **Main 3-panel MP4**: `results/teleop_hand_yolo_depth_robot.mp4`
- **GIF preview(slowed)**:

![Task 2 preview](results/teleop_hand_yolo_depth_robot.gif)

- **Verification MP4**: `results/teleop_conda_verify.mp4`
- **Smoke-test MP4 (ran inside conda)**: `results/teleop_conda_smoke.mp4`
