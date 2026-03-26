## Task 1 (SmolVLA + LeRobot) — run + results

### Conda environment (name: `lerobot-task1`)

From inside `submission/task1/`:

```bash
conda create -n lerobot-task1 python=3.10 -y
conda activate lerobot-task1

# Install LeRobot and its dependencies (GPU is optional; follow PyTorch CUDA docs if needed).
pip install lerobot
pip install huggingface_hub
```

If your install is missing MuJoCo / Gym env dependencies for ALOHA, install what the error asks for
(common extras are `mujoco`, `gymnasium`, and the ALOHA env package used by LeRobot).

### Run

From inside `submission/task1/` (with env activated):

If you did not commit the large checkpoint weights to git, download them first:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='c27e/smolvla_aloha_sim_insertion_human', local_dir='model/smolvla_aloha_sim_insertion_human', local_dir_use_symlinks=False)"
```

```bash
python run_partb_smolvla_aloha_insertion.py \
  --model-path model/smolvla_aloha_sim_insertion_human \
  --seed 1 \
  --max-steps 500 \
  --save-video results/task1_partb_seed1_prompted_500.mp4
```

### Output

- **Video**: `results/task1_partb_seed1_prompted_500.mp4`
- **GIF preview**:

![Task 1 preview](results/task1_partb_seed1_prompted_500.gif)

- **Smoke-test video (ran inside conda)**: `results/task1_conda_smoke.mp4`
- **Reports**:
  - `reports/task1-part-A.pdf`
  - `reports/task-1-report.pdf`

