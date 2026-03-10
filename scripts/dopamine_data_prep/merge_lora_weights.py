#!/usr/bin/env python3
"""
Merge LoRA adapter weights into the base model for vLLM inference.

Usage:
    python merge_lora_weights.py \
        --base_model tanhuajie2001/Robo-Dopamine-GRM-3B \
        --lora_path ./checkpoints/mimicgen_grm_lora \
        --output_path ./checkpoints/mimicgen_grm_merged
"""

import argparse
import torch
from pathlib import Path


def merge_lora(base_model: str, lora_path: str, output_path: str):
    from peft import PeftModel
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
    )

    print(f"Loading base model from {base_model}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {lora_path}...")
    model = PeftModel.from_pretrained(model, lora_path)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))

    # Also save processor/tokenizer for vLLM
    print("Saving processor...")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    processor.save_pretrained(str(output_dir))

    print(f"Merged model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="tanhuajie2001/Robo-Dopamine-GRM-3B",
        help="Base model name or path",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for merged model",
    )
    args = parser.parse_args()
    merge_lora(args.base_model, args.lora_path, args.output_path)


if __name__ == "__main__":
    main()
