#!/usr/bin/env python3
"""
Subprocess worker for Robo-Dopamine dense reward computation.

This script runs in the `robo-dopamine` conda environment (torch 2.8 + vLLM),
separate from the RL training process (equidiff env, torch 2.1).

Communication protocol:
  1. Parent saves rl_data to a .npz file
  2. Parent launches this worker via subprocess
  3. Worker loads vLLM, computes rewards, saves to output .npy file
  4. Parent reads the output

Usage (called by DopamineRewardModel, not directly):
    conda run -n robo-dopamine python _reward_worker.py \
        --input_path /tmp/dopamine_input.npz \
        --output_path /tmp/dopamine_output.npy \
        --model_path ./checkpoints/three_piece_grm_merged \
        --task "assemble three pieces" \
        --goal_high_path /tmp/goal_high.npy \
        --goal_wrist_path /tmp/goal_wrist.npy \
        --gamma 0.99 \
        --perspectives incremental,forward,backward \
        --dense_reward_scale 1.0 \
        --batch_size 4 \
        --gpu_memory_utilization 0.85
"""

import argparse
import os
import sys
import re
import numpy as np
from typing import List, Tuple, Dict
from PIL import Image

os.environ['LOCAL_RANK'] = '0'


SYSTEM_PROMPT = """
You are a rigorous, impartial vision evaluator for robot task progress. Your job is to judge whether the AFTER image set moves closer to the task objective than the BEFORE image set, using the provided reference examples only as anchors.

<Task>
`{task}`

REFERENCE EXAMPLES (for visual anchoring only; not necessarily this run's actual START/END):
- REFERENCE START — Robot Front Image (task just starting): <image>
- REFERENCE END — Robot Front Image (task fully completed): <image>
</Task>

BEFORE Robot Front Image: <image>
BEFORE Robot Left Wrist Image: <image>
BEFORE Robot Right Wrist Image: <image>

AFTER Robot Front Image: <image>
AFTER Robot Left Wrist Image: <image>
AFTER Robot Right Wrist Image: <image>

Goal
Compare the BEFORE and AFTER three-view sets and judge whether AFTER moves closer to accomplishing the task than BEFORE, using the REFERENCE START/END images as conceptual anchors.

Progress Estimation (no formulas)
1) Calibrate using the references:
   - REFERENCE START = "just beginning"; REFERENCE END = "fully completed."
   - Visually estimate how far BEFORE and AFTER are along this START→END continuum.
2) Direction:
   - AFTER better than BEFORE → positive score.
   - AFTER worse than BEFORE → negative score.
   - Essentially the same → 0.
3) Normalize to an integer percentage in [-100%, +100%]:
   - For improvements, scale the improvement relative to what remained from BEFORE to END.
   - For regressions, scale the deterioration relative to how far BEFORE had progressed from START.
   - Clip to [-100%, +100%] and round to the nearest integer percent.

Evaluation Criteria (apply across all three views)
1) Task Alignment: Evidence directly tied to `{task}`.
2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.
3) View-Specific Evidence & Consistency:
   - Use the **Front** view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.
   - Use the **Left/Right Wrist** views to inspect **fine-grained gripper state** (finger closure, contact location/area, slippage, wedge/misalignment, object deformation, cable/wire/cloth entanglement, unintended contact, occluded collisions).
   - When views disagree, prioritize the view that provides **decisive cues** for the criterion at hand. In particular, wrist views often **override** for grasp/contact validity and safety.
   - If any single view shows a failure that invalidates success (e.g., mis-grasp, collision, unsafe/unstable pose), let that override when judging progress.
4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that don't affect task success.
5) Ambiguity: If evidence is genuinely inconclusive or conflicting without decisive cues, treat progress as unchanged → 0%.

Output Format (STRICT)
Return ONLY one line containing the score wrapped in <score> tags, as an integer percentage with a percent sign:
<score>+NN%</score>  or  <score>-NN%</score>  or  <score>0%</score>
"""


def parse_score(text: str) -> float:
    try:
        val_str = text.split("<score>")[-1].split("</score>")[0].replace("%", "").strip()
        return max(-100.0, min(100.0, float(val_str))) / 100.0
    except Exception:
        return 0.0


def numpy_to_pil(img: np.ndarray, size: Tuple[int, int] = (224, 224)) -> Image.Image:
    if img.dtype in (np.float32, np.float64):
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    else:
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)
    pil = Image.fromarray(img, mode="RGB")
    if size:
        resample = getattr(Image, 'Resampling', Image).BILINEAR
        pil = pil.resize(size, resample)
    return pil


def batch_vlm_inference(model, processor, sampling_params, task: str,
                        all_items: List[List[Image.Image]], batch_size: int) -> np.ndarray:
    """Run batched VLM inference. Returns scores array [-1, 1]."""
    prompt_template = SYSTEM_PROMPT.format(task=task)
    prompt_parts = prompt_template.split("<image>")
    all_scores = []

    for start in range(0, len(all_items), batch_size):
        batch = all_items[start:start + batch_size]
        prompts = []
        for images in batch:
            content = []
            for i, part in enumerate(prompt_parts):
                if part:
                    content.append({"type": "text", "text": part})
                if i < len(prompt_parts) - 1:
                    content.append({"type": "image"})
            messages = [{"role": "user", "content": content}]
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append({"prompt": prompt_text, "multi_modal_data": {"image": images}})

        outputs = model.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        for out in outputs:
            all_scores.append(parse_score(out.outputs[0].text))

        if start > 0 and start % (batch_size * 10) == 0:
            print(f"  VLM progress: {start}/{len(all_items)}")

    return np.array(all_scores, dtype=np.float64)


def compute_perspective(mode: str, model, processor, sampling_params, task: str,
                        before_high, before_wrist, after_high, after_wrist,
                        init_high_pils, init_wrist_pils,
                        goal_high_pil, goal_wrist_pil,
                        n_steps, n_envs, img_size, batch_size) -> np.ndarray:
    """Compute progress array [T+1, B] for one perspective."""
    phi = np.zeros((n_steps + 1, n_envs))
    all_items = []

    for t in range(n_steps):
        for env in range(n_envs):
            af_high_pil = numpy_to_pil(after_high[t, env], img_size)
            af_wrist_pil = numpy_to_pil(after_wrist[t, env], img_size)

            if mode == "incremental":
                bf_high_pil = numpy_to_pil(before_high[t, env], img_size)
                bf_wrist_pil = numpy_to_pil(before_wrist[t, env], img_size)
            elif mode == "forward":
                bf_high_pil = init_high_pils[env]
                bf_wrist_pil = init_wrist_pils[env]
            elif mode == "backward":
                bf_high_pil = goal_high_pil
                bf_wrist_pil = goal_wrist_pil

            images = [
                init_high_pils[env], goal_high_pil,
                bf_high_pil, bf_wrist_pil, bf_wrist_pil,
                af_high_pil, af_wrist_pil, af_wrist_pil,
            ]
            all_items.append(images)

    scores = batch_vlm_inference(model, processor, sampling_params, task, all_items, batch_size)
    scores = scores.reshape(n_steps, n_envs)

    if mode == "incremental":
        for t in range(n_steps):
            for env in range(n_envs):
                H = scores[t, env]
                if t == 0:
                    phi[1, env] = H
                else:
                    prev = phi[t, env]
                    phi[t + 1, env] = prev + H * ((1 - prev) if H >= 0 else prev)
    elif mode == "forward":
        for t in range(n_steps):
            phi[t + 1] = scores[t]
    elif mode == "backward":
        for t in range(n_steps):
            phi[t + 1] = 1.0 + scores[t]

    return phi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--goal_high_path", required=True)
    parser.add_argument("--goal_wrist_path", required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--perspectives", default="incremental,forward,backward")
    parser.add_argument("--dense_reward_scale", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--image_resize", type=int, nargs=2, default=[224, 224])
    args = parser.parse_args()

    perspectives = [p.strip() for p in args.perspectives.split(",")]
    img_size = tuple(args.image_resize)

    print(f"[reward_worker] Loading input from {args.input_path}")
    data = np.load(args.input_path, allow_pickle=False)
    before_high = data["before_high"]    # [T, B, C, H, W] or [T, B, H, W, C]
    before_wrist = data["before_wrist"]
    after_high = data["after_high"]
    after_wrist = data["after_wrist"]
    n_steps, n_envs = before_high.shape[:2]
    print(f"[reward_worker] Data: n_steps={n_steps}, n_envs={n_envs}, shape={before_high.shape}")

    goal_high = np.load(args.goal_high_path)
    goal_wrist = np.load(args.goal_wrist_path)
    goal_high_pil = numpy_to_pil(goal_high, img_size)
    goal_wrist_pil = numpy_to_pil(goal_wrist, img_size)

    init_high_pils = [numpy_to_pil(before_high[0, env], img_size) for env in range(n_envs)]
    init_wrist_pils = [numpy_to_pil(before_wrist[0, env], img_size) for env in range(n_envs)]

    # Load vLLM
    print(f"[reward_worker] Loading vLLM model from {args.model_path}")
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    model = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 8},
        enable_prefix_caching=True,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, top_k=50, max_tokens=1024)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = 76800
        processor.image_processor.min_pixels = 12544
    print(f"[reward_worker] Model loaded.")

    # Compute each perspective
    phi_dict = {}
    for mode in perspectives:
        print(f"[reward_worker] Computing {mode} perspective...")
        phi_dict[mode] = compute_perspective(
            mode, model, processor, sampling_params, args.task,
            before_high, before_wrist, after_high, after_wrist,
            init_high_pils, init_wrist_pils,
            goal_high_pil, goal_wrist_pil,
            n_steps, n_envs, img_size, args.batch_size
        )

    # Fuse
    phi_fused = np.zeros((n_steps + 1, n_envs))
    for phi in phi_dict.values():
        phi_fused += phi
    phi_fused /= len(phi_dict)

    # Dense reward
    dense_rewards = np.zeros((n_steps, n_envs, 1))
    for t in range(n_steps):
        dense_rewards[t, :, 0] = args.gamma * phi_fused[t + 1] - phi_fused[t]
    dense_rewards *= args.dense_reward_scale

    print(f"[reward_worker] Dense reward stats: mean={dense_rewards.mean():.4f}, "
          f"std={dense_rewards.std():.4f}, min={dense_rewards.min():.4f}, max={dense_rewards.max():.4f}")
    for name, phi in phi_dict.items():
        print(f"  Phi_{name} final: mean={phi[-1].mean():.4f}")

    np.save(args.output_path, dense_rewards)
    print(f"[reward_worker] Saved rewards to {args.output_path}")


if __name__ == "__main__":
    main()
