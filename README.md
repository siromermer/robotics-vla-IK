## Task 1 + Task 2 (Embodied AI)

This repository contains two tasks:

- **Task 1**: SmolVLA + LeRobot policy inference in MuJoCo (ALOHA insertion)
- **Task 2**: Vision-based teleoperation (hand video → VX300s right arm) with from-scratch IK in MuJoCo.
  A 7-step pipeline (MediaPipe hand tracking → YOLOv8n cup detection → DepthAnythingV2 depth →
  trajectory mapping → damped least-squares IK → MuJoCo simulation → 3-panel MP4) that converts
  raw RGB video of a hand holding a cup into joint-angle commands for the simulated robot arm,
  with no pre-recorded robot demonstrations required.
  See [`task2/README.md`](task2/README.md) for a full component-by-component explanation.

### Folder structure

- **`task1/`**: code + reports + result video
- **`task2/`**: code + assets + result video(s)

Each task folder has its own `README.md` with **conda env setup** and **how to run**.

### Results

- **Task 1 (SmolVLA inference) video**: [`task1/results/task1_partb_seed1_prompted_500.mp4`](task1/results/task1_partb_seed1_prompted_500.mp4)
- **Task 2 (Teleop 3-panel) video**: [`task2/results/teleop_hand_yolo_depth_robot.mp4`](task2/results/teleop_hand_yolo_depth_robot.mp4)

#### Task 1 preview (GIF, slowed)

![Task 1 preview](task1/results/task1_partb_seed1_prompted_500.gif)

#### Task 2 preview (GIF, slowed)

![Task 2 preview](task2/results/teleop_hand_yolo_depth_robot.gif)

### Reports (Task 1)

- Task 1 Part A report: [`task1/reports/task1-part-A.pdf`](task1/reports/task1-part-A.pdf)
- Task 1 Part B report : [`task1/reports/task-1-report.pdf`](task1/reports/task-1-report.pdf)

