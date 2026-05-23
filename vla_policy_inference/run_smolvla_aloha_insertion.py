#!/usr/bin/env python3
"""Run SmolVLA inference for ALOHA simulated bimanual peg insertion.

This script is tailored for:
- model: c27e/smolvla_aloha_sim_insertion_human
- env task: AlohaInsertion-v0 (gym_aloha / MuJoCo)
- camera key: observation.images.top (from env pixels/top)
- state key: observation.state (14D)
- action key: action (14D)
- action chunking: chunk_size=50, n_action_steps=50

It explicitly queries the policy for a 50-step trajectory and executes all
50 actions sequentially before querying the model again.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import AlohaEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "model"
    / "smolvla_aloha_sim_insertion_human"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolVLA ALOHA insertion inference")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Local path or HF repo ID for the pretrained policy.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="AlohaInsertion-v0",
        help="ALOHA simulation task name.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Maximum simulator steps for one episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Environment seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default="",
        help="Optional MP4 output path. If empty, no video is saved.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Simulation and output video FPS.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Insert the peg into the socket.",
        help=(
            "Language instruction passed to SmolVLA. "
            "Use the same wording as the training dataset when possible."
        ),
    )
    return parser.parse_args()


def pick_device(arg_device: str) -> str:
    if arg_device == "auto":
        # Default to CUDA when available so inference is fast enough for video capture.
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg_device


def main() -> None:
    args = parse_args()
    model_path = args.model_path
    device = pick_device(args.device)
    # model_path can be a local folder (downloaded checkpoint) or an HF repo id.

    # Build ALOHA environment config matching training setup.
    # obs_type='pixels_agent_pos' provides:
    # - pixels/top -> observation.images.top
    # - agent_pos   -> observation.state
    env_cfg = AlohaEnv(
        task=args.task,
        fps=args.fps,
        obs_type="pixels_agent_pos",
        observation_height=480,
        observation_width=640,
        render_mode="rgb_array",
        episode_length=args.max_steps,
    )

    # Load policy config + policy from checkpoint.
    policy_cfg = PreTrainedConfig.from_pretrained(model_path)
    policy_cfg.pretrained_path = model_path
    policy_cfg.device = device
    policy_cfg.use_amp = False
    # Force deterministic fp32 behavior here; AMP can change outputs and complicate debugging.

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    policy.reset()
    # reset() is important for policies with internal state (e.g., action chunking buffers).

    # Load policy pre/post processors from checkpoint files.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=model_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    env_preprocessor, _env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
    )

    # Create a single vectorized env instance.
    envs = make_env(env_cfg, n_envs=1)
    env = envs["aloha"][0]
    # LeRobot's factory returns a dict of task families; we use the first ALOHA env.

    # Reset and validate required observation structure.
    obs, info = env.reset(seed=args.seed)
    if "agent_pos" not in obs:
        raise RuntimeError("Expected 'agent_pos' in env observation.")
    if obs["agent_pos"].shape[-1] != 14:
        raise RuntimeError(f"Expected 14D state, got shape {obs['agent_pos'].shape}.")

    if "pixels" not in obs or "top" not in obs["pixels"]:
        raise RuntimeError("Expected top camera at observation key pixels/top.")
    if obs["pixels"]["top"].shape[-3:] != (480, 640, 3):
        raise RuntimeError(
            f"Expected top camera shape (480,640,3), got {obs['pixels']['top'].shape}."
        )

    chunk_size = int(policy_cfg.chunk_size)
    n_action_steps = int(policy_cfg.n_action_steps)
    if chunk_size != 50 or n_action_steps != 50:
        raise RuntimeError(
            f"Expected chunk_size=50 and n_action_steps=50, got {chunk_size} and {n_action_steps}."
        )

    frames: list[np.ndarray] = []
    total_reward = 0.0
    done = False
    step = 0
    n_chunks = 0

    print(f"Running policy on device={device}")
    print(f"Task={args.task}, max_steps={args.max_steps}, chunk_size={chunk_size}")
    print(f"VLA prompt={args.prompt!r}")

    while not done and step < args.max_steps:
        # Convert env obs -> policy input format.
        proc_obs: dict[str, Any] = preprocess_observation(obs)
        proc_obs = add_envs_task(env, proc_obs)
        # Force a known instruction text for SmolVLA instead of relying on the
        # env's short task token (e.g. 'insertion').
        proc_obs["task"] = [args.prompt]
        proc_obs = env_preprocessor(proc_obs)
        proc_obs = preprocessor(proc_obs)
        # After preprocessing, proc_obs contains tensors ready for policy inference.

        # Query model ONCE for a 50-step action trajectory.
        with torch.inference_mode():
            action_chunk = policy.predict_action_chunk(proc_obs)  # [B=1, T=50, A=14]

        if action_chunk.ndim != 3:
            raise RuntimeError(f"Expected chunk tensor rank 3, got {action_chunk.shape}.")
        if action_chunk.shape != (1, 50, 14):
            raise RuntimeError(f"Expected action chunk shape (1,50,14), got {tuple(action_chunk.shape)}.")

        # Convert back to action space expected by the simulator.
        action_seq = postprocessor(action_chunk[0])  # [50, 14], usually moved to CPU
        if action_seq.shape != (50, 14):
            raise RuntimeError(f"Expected postprocessed actions (50,14), got {tuple(action_seq.shape)}.")

        n_chunks += 1

        # Execute the full 50-step action chunk sequentially in MuJoCo.
        for i in range(50):
            action_step = action_seq[i].detach().cpu().numpy().astype(np.float32)
            # Env is vectorized: step expects shape (B, A), so we add a batch dimension.
            obs, reward, terminated, truncated, info = env.step(action_step[None, :])
            total_reward += float(reward[0])
            step += 1

            if args.save_video:
                # Render returns an RGB array; we collect frames and encode at the end.
                frames.append(env.envs[0].render())

            done = bool(terminated[0] or truncated[0])
            if done or step >= args.max_steps:
                break

    env.close()

    print("Inference summary")
    print(f"steps_executed={step}")
    print(f"policy_queries={n_chunks}")
    print(f"total_reward={total_reward:.4f}")
    print(f"terminated={done}")

    if args.save_video:
        out_path = Path(args.save_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio

            if len(frames) == 0:
                print("No frames collected; skipping video save.")
            else:
                # Save as MP4/GIF depending on extension; imageio selects writer automatically.
                imageio.mimsave(out_path, frames, fps=args.fps)
                print(f"Saved video: {out_path}")
        except Exception as exc:  # pragma: no cover
            print(f"Failed to save video at {out_path}: {exc}")


if __name__ == "__main__":
    main()
