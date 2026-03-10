#!/usr/bin/env python3
"""
Extract demo frames from MimicGen HDF5 datasets for Robo-Dopamine training.

Converts MimicGen HDF5 format to the directory structure expected by
Robo-Dopamine's data processing pipeline:
  output_dir/
    task_instruction.json
    episode_000/
      cam_high/frame_000000.jpg, frame_000001.jpg, ...
      cam_left_wrist/frame_000000.jpg, ...
      cam_right_wrist/frame_000000.jpg, ...
    episode_001/
      ...
    goal_images/
      goal_000.png, goal_001.png, ...

Usage:
    python extract_mimicgen_demos.py \
        --dataset_path path/to/dataset.hdf5 \
        --output_dir path/to/output \
        --task_description "stack three blocks" \
        --max_episodes 100
"""

import os
import argparse
import json
import numpy as np
import h5py
import cv2
from pathlib import Path
from tqdm import tqdm


def extract_demos(
    dataset_path: str,
    output_dir: str,
    task_description: str = "perform the task",
    max_episodes: int = None,
    image_keys: list = None,
):
    """
    Extract demo frames from MimicGen HDF5 dataset.

    Args:
        dataset_path: Path to HDF5 dataset
        output_dir: Output directory for extracted frames
        task_description: Task description for Robo-Dopamine
        max_episodes: Maximum number of episodes to extract (None = all)
        image_keys: List of image observation keys to extract
    """
    if image_keys is None:
        image_keys = ["agentview_image", "robot0_eye_in_hand_image"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    goal_dir = output_dir / "goal_images"
    goal_dir.mkdir(parents=True, exist_ok=True)

    # Save task instruction
    instruction_path = output_dir / "task_instruction.json"
    with open(instruction_path, "w") as f:
        json.dump([task_description], f, indent=2)
    print(f"Saved task instruction to {instruction_path}")

    # Open HDF5 dataset
    with h5py.File(dataset_path, "r") as hdf5_file:
        demos = hdf5_file["data"]
        demo_keys = sorted(demos.keys(), key=lambda x: int(x.replace("demo_", "")))

        if max_episodes is not None:
            demo_keys = demo_keys[:max_episodes]

        print(f"Extracting {len(demo_keys)} episodes from {dataset_path}")

        # Check available obs keys
        sample_demo = demos[demo_keys[0]]
        available_obs_keys = list(sample_demo["obs"].keys())
        print(f"Available obs keys: {available_obs_keys}")

        # Validate requested keys exist
        for key in image_keys:
            if key not in available_obs_keys:
                raise ValueError(
                    f"Image key '{key}' not found. Available: {available_obs_keys}"
                )

        # Check image shape
        sample_img = sample_demo["obs"][image_keys[0]][0]
        print(f"Sample image shape: {sample_img.shape}, dtype: {sample_img.dtype}")

        for ep_idx, demo_key in enumerate(tqdm(demo_keys, desc="Extracting episodes")):
            demo = demos[demo_key]
            ep_name = f"episode_{ep_idx:03d}"
            ep_dir = output_dir / ep_name

            # Camera mapping:
            # agentview_image -> cam_high
            # robot0_eye_in_hand_image -> cam_left_wrist AND cam_right_wrist
            cam_dirs = {
                "cam_high": ep_dir / "cam_high",
                "cam_left_wrist": ep_dir / "cam_left_wrist",
                "cam_right_wrist": ep_dir / "cam_right_wrist",
            }
            for d in cam_dirs.values():
                d.mkdir(parents=True, exist_ok=True)

            # Extract agentview_image -> cam_high
            agentview_imgs = demo["obs"]["agentview_image"][:]
            n_frames = len(agentview_imgs)

            for frame_idx in range(n_frames):
                img = agentview_imgs[frame_idx]
                # HDF5 images are typically uint8 RGB [H, W, 3]
                # or float [C, H, W] - handle both
                img = _normalize_image(img)

                frame_path = cam_dirs["cam_high"] / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Extract robot0_eye_in_hand_image -> cam_left_wrist AND cam_right_wrist
            eye_in_hand_imgs = demo["obs"]["robot0_eye_in_hand_image"][:]

            for frame_idx in range(n_frames):
                img = eye_in_hand_imgs[frame_idx]
                img = _normalize_image(img)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Same image for both wrist cameras (only one wrist camera available)
                for cam_name in ["cam_left_wrist", "cam_right_wrist"]:
                    frame_path = cam_dirs[cam_name] / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_path), img_bgr)

            # Extract goal image (last frame of agentview)
            goal_img = agentview_imgs[-1]
            goal_img = _normalize_image(goal_img)
            goal_path = goal_dir / f"goal_{ep_idx:03d}.png"
            cv2.imwrite(str(goal_path), cv2.cvtColor(goal_img, cv2.COLOR_RGB2BGR))

    # Create a single representative goal image (from first episode)
    representative_goal = goal_dir / "goal_000.png"
    if representative_goal.exists():
        default_goal = output_dir / "goal_image.png"
        import shutil
        shutil.copy(str(representative_goal), str(default_goal))
        print(f"Created default goal image at {default_goal}")

    print(f"\nExtraction complete!")
    print(f"  Episodes: {len(demo_keys)}")
    print(f"  Output: {output_dir}")
    print(f"  Goal images: {goal_dir}")


def _normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 RGB [H, W, 3]."""
    if img.dtype == np.float32 or img.dtype == np.float64:
        # Float images: check if [C, H, W] or [H, W, C]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))  # [C, H, W] -> [H, W, C]
        # Scale to [0, 255]
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    elif img.dtype == np.uint8:
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
    else:
        img = img.astype(np.uint8)

    # Handle grayscale
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)

    return img


def main():
    parser = argparse.ArgumentParser(
        description="Extract MimicGen demo frames for Robo-Dopamine training"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to MimicGen HDF5 dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for extracted frames",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="perform the task",
        help="Task description for Robo-Dopamine",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to extract (default: all)",
    )

    args = parser.parse_args()

    extract_demos(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        task_description=args.task_description,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
