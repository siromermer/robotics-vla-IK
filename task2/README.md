## Task 2 (hand video → VX300s right arm) — conda + run

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
