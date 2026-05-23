# Robotics VLA and Vision Teleoperation

Robotics project combining vision-language-action policy inference, robot
simulation, hand-object perception, and inverse kinematics.

## Components

- `vla_policy_inference/`: SmolVLA inference in the LeRobot ALOHA insertion environment.
- `vision_teleoperation/`: Hand-video-to-VX300s teleoperation pipeline with MediaPipe, YOLO, Depth Anything V2, MuJoCo, and damped least-squares IK.

## Results

- Video: [`vla_policy_inference/results/smolvla_aloha_insertion_seed1_500.mp4`](vla_policy_inference/results/smolvla_aloha_insertion_seed1_500.mp4)
- Video: [`vision_teleoperation/results/teleop_hand_yolo_depth_robot.mp4`](vision_teleoperation/results/teleop_hand_yolo_depth_robot.mp4)

#### SmolVLA ALOHA insertion preview (GIF, slowed)

![SmolVLA ALOHA insertion preview](vla_policy_inference/results/smolvla_aloha_insertion_seed1_500.gif)

#### Vision teleoperation preview (GIF, slowed)

![Vision teleoperation preview](vision_teleoperation/results/teleop_hand_yolo_depth_robot.gif)

## Quick Start

Create separate environments for the two components because the VLA policy
stack and the teleoperation stack have different dependency profiles.

### SmolVLA Inference

```bash
conda create -n robotics-vla python=3.10 -y
conda activate robotics-vla
pip install lerobot huggingface_hub imageio[ffmpeg]

python vla_policy_inference/run_smolvla_aloha_insertion.py \
  --seed 1 \
  --max-steps 500 \
  --save-video vla_policy_inference/results/smolvla_aloha_insertion_seed1_500.mp4
```

### Vision Teleoperation

```bash
conda create -n robotics-teleop python=3.10 -y
conda activate robotics-teleop
pip install -r vision_teleoperation/requirements.txt

python -m vision_teleoperation.teleop_main \
  --video vision_teleoperation/test_video.mp4 \
  --output vision_teleoperation/results/teleop_hand_yolo_depth_robot.mp4
```

## Implementation Notes

- VLA policy: `c27e/smolvla_aloha_sim_insertion_human`.
- Simulation backends: MuJoCo ALOHA and Trossen VX300s MJCF.
- Teleoperation perception: MediaPipe hand landmarks, YOLOv8n cup detection, Depth Anything V2 relative depth.
- IK method: from-scratch geometric Jacobian with damped least-squares updates.

## Documentation

- [SmolVLA inference](vla_policy_inference/README.md)
- [Vision teleoperation](vision_teleoperation/README.md)
