#!/usr/bin/env python3
"""
Robo-Dopamine Dense Reward Model for RL Training (subprocess mode).

Since the RL training (equidiff env, torch 2.1) and reward model (robo-dopamine env,
torch 2.8 + vLLM) have incompatible dependencies, this wrapper communicates via
subprocess + temporary files:

  1. RL process saves observation images to a temp .npz file
  2. Launches _reward_worker.py in the robo-dopamine conda environment
  3. Worker loads vLLM, computes dense rewards, saves to .npy file
  4. RL process reads back the dense rewards

The reward model is frozen — no gradients, no fine-tuning during RL.
"""

import os
import subprocess
import tempfile
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


def _numpy_to_uint8(img_array: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 for saving. Handles [C,H,W] float and [H,W,C] uint8."""
    if img_array.dtype in (np.float32, np.float64):
        if img_array.ndim == 3 and img_array.shape[0] in (1, 3):
            img_array = np.transpose(img_array, (1, 2, 0))
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        else:
            img_array = img_array.clip(0, 255).astype(np.uint8)
    elif img_array.dtype != np.uint8:
        if img_array.ndim == 3 and img_array.shape[0] in (1, 3):
            img_array = np.transpose(img_array, (1, 2, 0))
        img_array = img_array.astype(np.uint8)
    else:
        if img_array.ndim == 3 and img_array.shape[0] in (1, 3):
            img_array = np.transpose(img_array, (1, 2, 0))
    return img_array


class DopamineRewardModel:
    """
    Dense reward model using Robo-Dopamine VLM via subprocess.

    Runs VLM inference in a separate conda environment (robo-dopamine) to avoid
    dependency conflicts with the RL training environment (equidiff).
    The model is frozen — only provides dense reward signal.
    """

    def __init__(
        self,
        model_path: str,
        task_description: str,
        goal_images: Dict[str, np.ndarray],
        gamma: float = 0.99,
        perspectives: List[str] = None,
        sparse_reward_bonus: float = 10.0,
        dense_reward_scale: float = 1.0,
        gpu_memory_utilization: float = 0.85,
        inference_batch_size: int = 4,
        image_resize: Tuple[int, int] = (224, 224),
        conda_env: str = "robo-dopamine",
        worker_script: str = None,
    ):
        self.model_path = model_path
        self.task = task_description
        self.gamma = gamma
        self.perspectives = perspectives or ["incremental", "forward", "backward"]
        self.sparse_reward_bonus = sparse_reward_bonus
        self.dense_reward_scale = dense_reward_scale
        self.gpu_memory_utilization = gpu_memory_utilization
        self.inference_batch_size = inference_batch_size
        self.image_resize = image_resize
        self.conda_env = conda_env

        # Worker script path
        if worker_script is None:
            self.worker_script = str(
                Path(__file__).parent / "_reward_worker.py"
            )
        else:
            self.worker_script = worker_script

        # Goal images — save to temp files for worker
        self._tmp_dir = tempfile.mkdtemp(prefix="dopamine_reward_")
        self._goal_high_path = os.path.join(self._tmp_dir, "goal_high.npy")
        self._goal_wrist_path = os.path.join(self._tmp_dir, "goal_wrist.npy")
        np.save(self._goal_high_path, goal_images["agentview"])
        np.save(self._goal_wrist_path, goal_images["eye_in_hand"])

        print(f"DopamineRewardModel initialized (subprocess mode):")
        print(f"  model_path: {model_path}")
        print(f"  task: {task_description}")
        print(f"  perspectives: {self.perspectives}")
        print(f"  conda_env: {conda_env}")
        print(f"  worker: {self.worker_script}")
        print(f"  gamma: {gamma}, sparse_bonus: {sparse_reward_bonus}")

    def load_model(self):
        """No-op: model is loaded inside the subprocess."""
        pass

    def unload_model(self):
        """No-op: subprocess exits and releases GPU on its own."""
        pass

    def compute_rewards(self, rl_data: Dict[str, Any]) -> np.ndarray:
        """
        Compute dense rewards by launching a subprocess in the robo-dopamine env.

        Args:
            rl_data: Dict with observations, after_observations, total_steps, total_envs

        Returns:
            dense_rewards: np.ndarray [n_steps, n_envs, 1]
        """
        n_steps = rl_data["total_steps"]
        n_envs = rl_data["total_envs"]

        # Extract images (take last obs_horizon frame)
        before_high = rl_data["observations"]["agentview_image"][:, :, -1]
        before_wrist = rl_data["observations"]["robot0_eye_in_hand_image"][:, :, -1]
        after_high = rl_data["after_observations"]["agentview_image"][:, :, -1]
        after_wrist = rl_data["after_observations"]["robot0_eye_in_hand_image"][:, :, -1]

        # Save to temp file
        input_path = os.path.join(self._tmp_dir, "input.npz")
        output_path = os.path.join(self._tmp_dir, "output.npy")

        # Remove old output if exists
        if os.path.exists(output_path):
            os.remove(output_path)

        np.savez(
            input_path,
            before_high=before_high,
            before_wrist=before_wrist,
            after_high=after_high,
            after_wrist=after_wrist,
        )
        print(f"Saved reward input: {input_path} "
              f"(shape: [{n_steps}, {n_envs}, ...])")

        # Build subprocess command
        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python", "-u", self.worker_script,
            "--input_path", input_path,
            "--output_path", output_path,
            "--model_path", self.model_path,
            "--task", self.task,
            "--goal_high_path", self._goal_high_path,
            "--goal_wrist_path", self._goal_wrist_path,
            "--gamma", str(self.gamma),
            "--perspectives", ",".join(self.perspectives),
            "--dense_reward_scale", str(self.dense_reward_scale),
            "--batch_size", str(self.inference_batch_size),
            "--gpu_memory_utilization", str(self.gpu_memory_utilization),
            "--image_resize", str(self.image_resize[0]), str(self.image_resize[1]),
        ]

        print(f"Launching reward worker subprocess...")
        start_time = time.time()

        result = subprocess.run(
            cmd,
            capture_output=False,  # Let output go to parent's stdout/stderr
            text=True,
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"ERROR: Reward worker failed (exit code {result.returncode})")
            print(f"Returning zero dense rewards as fallback.")
            return np.zeros((n_steps, n_envs, 1))

        # Read results
        if not os.path.exists(output_path):
            print(f"ERROR: Output file not found: {output_path}")
            return np.zeros((n_steps, n_envs, 1))

        dense_rewards = np.load(output_path)
        print(f"Dense rewards computed in {elapsed:.1f}s: "
              f"mean={dense_rewards.mean():.4f}, std={dense_rewards.std():.4f}")

        return dense_rewards

    def __del__(self):
        """Cleanup temp files."""
        import shutil
        if hasattr(self, '_tmp_dir') and os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)


def load_goal_images_from_demos(dataset_path: str) -> Dict[str, np.ndarray]:
    """
    Load goal images from a MimicGen HDF5 dataset.
    Uses the last frame of the first demo.
    """
    import h5py

    with h5py.File(dataset_path, "r") as f:
        demos = f["data"]
        demo_keys = sorted(demos.keys(), key=lambda x: int(x.replace("demo_", "")))
        demo = demos[demo_keys[0]]
        goal_agentview = demo["obs"]["agentview_image"][-1]
        goal_eye_in_hand = demo["obs"]["robot0_eye_in_hand_image"][-1]

    print(f"Loaded goal images from {dataset_path} (demo_0, last frame)")
    print(f"  agentview: {goal_agentview.shape}, eye_in_hand: {goal_eye_in_hand.shape}")

    return {
        "agentview": goal_agentview,
        "eye_in_hand": goal_eye_in_hand,
    }
