# Vision-Based VX300s Teleoperation

Maps a hand-object video to a VX300s end-effector trajectory, solves inverse
kinematics, replays the motion in MuJoCo, and exports a three-panel video.

## Pipeline

1. Detect hand landmarks with MediaPipe.
2. Detect the cup with YOLOv8n.
3. Estimate relative depth with Depth Anything V2.
4. Convert hand-object motion to end-effector targets.
5. Solve VX300s IK using a geometric Jacobian and damped least squares.
6. Render the simulated robot trajectory in MuJoCo.

## Environment

```bash
conda create -n robotics-teleop python=3.10 -y
conda activate robotics-teleop
pip install -r vision_teleoperation/requirements.txt
```

Use the command above from the repository root.

## Run

From the repository root:

```bash
python -m vision_teleoperation.teleop_main \
  --video vision_teleoperation/test_video.mp4 \
  --output vision_teleoperation/results/teleop_hand_yolo_depth_robot.mp4
```

On headless Linux with NVIDIA rendering, set MuJoCo EGL explicitly:

```bash
export MUJOCO_GL=egl
```

## Output

- MP4: [`results/teleop_hand_yolo_depth_robot.mp4`](results/teleop_hand_yolo_depth_robot.mp4)
- GIF: [`results/teleop_hand_yolo_depth_robot.gif`](results/teleop_hand_yolo_depth_robot.gif)

![Vision teleoperation preview](results/teleop_hand_yolo_depth_robot.gif)

## Main Files

- `teleop_main.py`: full perception, trajectory, IK, rendering, and video composition pipeline.
- `hand_tracker.py`: MediaPipe hand landmark extraction.
- `cup_detector.py`: YOLOv8n cup detection.
- `depth_estimator.py`: Depth Anything V2 relative depth maps.
- `vx300s_ik.py`: from-scratch forward kinematics, Jacobian, and damped least-squares IK.
- `assets/trossen_vx300s/`: VX300s MJCF model and mesh assets.
