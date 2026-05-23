# SmolVLA ALOHA Inference

Runs a pretrained SmolVLA policy in the LeRobot ALOHA insertion simulation and
optionally records the rollout.

## Environment

```bash
conda create -n robotics-vla python=3.10 -y
conda activate robotics-vla
pip install lerobot huggingface_hub imageio[ffmpeg]
```

If MuJoCo or ALOHA environment dependencies are missing, install the package
reported by the runtime error. Common packages are `mujoco` and `gymnasium`.

## Model Checkpoint

The repository contains the local model directory:

```text
model/smolvla_aloha_sim_insertion_human/
```

If the checkpoint files are absent, download them from Hugging Face:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='c27e/smolvla_aloha_sim_insertion_human', local_dir='model/smolvla_aloha_sim_insertion_human', local_dir_use_symlinks=False)"
```

## Run

From `vla_policy_inference/`:

```bash
python run_smolvla_aloha_insertion.py \
  --seed 1 \
  --max-steps 500 \
  --save-video results/smolvla_aloha_insertion_seed1_500.mp4
```

## Output

- MP4: [`results/smolvla_aloha_insertion_seed1_500.mp4`](results/smolvla_aloha_insertion_seed1_500.mp4)
- GIF: [`results/smolvla_aloha_insertion_seed1_500.gif`](results/smolvla_aloha_insertion_seed1_500.gif)

![SmolVLA ALOHA insertion preview](results/smolvla_aloha_insertion_seed1_500.gif)

## Main Files

- `run_smolvla_aloha_insertion.py`: policy loading, environment setup, rollout loop, and video export.
- `model/smolvla_aloha_sim_insertion_human/`: local pretrained policy files and processors.
- `results/`: generated rollout media.
