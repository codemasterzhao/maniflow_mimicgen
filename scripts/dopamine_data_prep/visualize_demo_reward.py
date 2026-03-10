#!/usr/bin/env python3
"""
Visualize Robo-Dopamine reward model on a MimicGen demo episode.

Usage:
    # With merged model:
    python scripts/visualize_demo_reward.py \
        --model_path ./checkpoints/three_piece_grm_merged \
        --demo_dir ./dataset/three_piece_assembly_raw/episode_000 \
        --goal_image ./dataset/three_piece_assembly_raw/goal_image.png \
        --task "assemble three pieces by placing each piece into the correct slot on the base" \
        --output_dir ./vis_results \
        --eval_mode forward

    # With base model (no fine-tuning):
    python scripts/visualize_demo_reward.py \
        --model_path tanhuajie2001/Robo-Dopamine-GRM-3B \
        ...
"""

import argparse
import os
import sys

# Import from local copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import GRMInference


def main():
    parser = argparse.ArgumentParser(description="Visualize reward model on a demo episode")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to merged GRM model or HF model ID")
    parser.add_argument("--demo_dir", type=str, required=True,
                        help="Path to demo episode directory (with cam_high/, cam_left_wrist/, cam_right_wrist/)")
    parser.add_argument("--goal_image", type=str, required=True,
                        help="Path to goal image")
    parser.add_argument("--task", type=str, default="assemble three pieces by placing each piece into the correct slot on the base",
                        help="Task description")
    parser.add_argument("--output_dir", type=str, default="./vis_results",
                        help="Output directory for visualization")
    parser.add_argument("--eval_mode", type=str, default="forward",
                        choices=["incremental", "forward", "backward"],
                        help="Evaluation mode")
    parser.add_argument("--frame_interval", type=int, default=10,
                        help="Frame sampling interval")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for VLM inference")

    args = parser.parse_args()

    # Validate demo directory
    cam_high = os.path.join(args.demo_dir, "cam_high")
    cam_left = os.path.join(args.demo_dir, "cam_left_wrist")
    cam_right = os.path.join(args.demo_dir, "cam_right_wrist")

    for d in [cam_high, cam_left, cam_right]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Camera directory not found: {d}")

    print(f"Model: {args.model_path}")
    print(f"Demo: {args.demo_dir}")
    print(f"Goal: {args.goal_image}")
    print(f"Mode: {args.eval_mode}")
    print(f"Frame interval: {args.frame_interval}")

    # Load model
    model = GRMInference(args.model_path)

    # Run pipeline
    output_dir = model.run_pipeline(
        cam_high_path=cam_high,
        cam_left_path=cam_left,
        cam_right_path=cam_right,
        out_root=args.output_dir,
        task=args.task,
        frame_interval=args.frame_interval,
        batch_size=args.batch_size,
        goal_image=args.goal_image,
        eval_mode=args.eval_mode,
        visualize=True,
    )

    print(f"\nVisualization complete!")
    print(f"Output directory: {output_dir}")
    print(f"Video: {os.path.join(output_dir, 'reward_vis.mp4')}")


if __name__ == "__main__":
    main()
