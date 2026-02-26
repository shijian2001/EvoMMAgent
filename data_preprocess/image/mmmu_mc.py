"""Preprocess MMMU dataset to JSONL + image folder format, **keeping only multiple-choice questions**.

This script is similar to `mmmu.py`, but filters out non-choice (e.g., open-ended/fill-in) questions.

Usage (from HuggingFace MMMU/MMMU, validation split):

  python mmmu_mc.py MMMU/MMMU \
      --jsonl_path /path/to/mmmu_val_mc.jsonl \
      --image_dir /path/to/mmmu_val_mc_images \
      --split validation

Example using local clone (same style as the original script):

  python mmmu_mc.py /mnt/sda/runhaofu/Datasets/MMMU/ \
      --jsonl_path /mnt/sda/runhaofu/Datasets/test_dataset/mmmu_mc.jsonl \
      --image_dir /mnt/sda/runhaofu/Datasets/test_dataset/mmmu_mc_images \
      --num_proc 8
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from mmmu import (  # type: ignore
    load_dataset,
    extract_images,
    save_images,
    _parse_options,
    _normalize_answer,
)


def is_multiple_choice(example: dict) -> bool:
    """Return True if this MMMU sample is a multiple-choice question.

    We primarily rely on `question_type` when available, and fall back to the
    presence of non-empty options.
    """
    qtype = (example.get("question_type") or "").lower()
    if qtype:
        # Common labels in MMMU for choice questions
        if qtype in {"multiple-choice", "multi-choice", "multiple_choice"}:
            return True
        if qtype in {"open", "open-ended", "fill-in", "free-form"}:
            return False

    # Fallback: treat samples with at least 2 parsed options as multiple-choice
    choices = _parse_options(example.get("options"))
    return len(choices) >= 2


def convert_sample_mc(example: dict, idx: int, image_paths: list[str]) -> dict:
    """Convert a *multiple-choice* MMMU sample to the unified BLINK-style record format.

    IMPORTANT: The output schema is aligned with BLINK:
    - Only these fields are kept: idx, images, dataset, type, sub_task,
      question, choices, answer, prompt.
    - Any field that BLINK有但当前样本没有的字段，统一置为空字符串。
    """
    options = example.get("options")
    choices = _parse_options(options)
    answer_raw = example.get("answer", "")
    answer = _normalize_answer(answer_raw, choices)

    record = {
        "idx": idx,
        "images": image_paths,
        "dataset": "MMMU",
        "type": "multi-choice",
        "sub_task": example.get("subfield") or example.get("topic_difficulty") or "",
        "question": example.get("question", ""),
        "choices": choices,
        "answer": answer,
        # BLINK 中存在但 MMMU 中没有的字段，统一置空
        "prompt": "",
    }
    return record


def process_and_save_mc(dataset, jsonl_path: Path, image_dir: Path):
    """Process dataset and save only multiple-choice records to JSONL + image folder.

    Notes:
    - `idx` is re-assigned to be dense [0..N_mc-1] over **kept** (MC) samples.
    - Image subfolders follow the same `{idx:05d}/{img_idx:05d}.png` pattern as `mmmu.py`.
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = len(dataset)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for orig_idx, example in tqdm(
            enumerate(dataset),
            total=total,
            desc="Processing MC samples",
        ):
            if not is_multiple_choice(example):
                continue

            images = extract_images(example)
            image_paths = save_images(images, kept, image_dir)
            record = convert_sample_mc(example, kept, image_paths)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"✅ Total samples in source dataset: {total}")
    print(f"✅ Kept multiple-choice samples: {kept}")
    print(f"✅ Saved MC JSONL to {jsonl_path}")
    print(f"✅ Saved MC images to {image_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MMMU to JSONL + image folder, keeping only multiple-choice questions",
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help='Path to MMMU dataset or HF name e.g. "MMMU/MMMU"',
    )
    parser.add_argument(
        "--jsonl_path",
        type=str,
        required=True,
        help="Output JSONL file path (multiple-choice only)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Output image folder path (multiple-choice only)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["dev", "validation", "test"],
        help="Dataset split to process (default: validation)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="(Unused here; kept for CLI compatibility with mmmu.py)",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.input_dir, split=args.split)
    process_and_save_mc(dataset, Path(args.jsonl_path), Path(args.image_dir))


if __name__ == "__main__":
    main()

