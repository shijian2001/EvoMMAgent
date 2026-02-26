"""Preprocess MMMU dataset to JSONL + image folder format, keeping only multiple-choice questions.

Processing pipeline:
1. load_dataset() - Load source dataset (reuses mmmu.py loader)
2. is_multiple_choice() - Filter to MC samples only
3. convert_sample_mc() - Convert sample to unified format
4. process_and_save_mc() - Coordinate saving images and JSONL
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
    """Return True if this MMMU sample is a multiple-choice question."""
    qtype = (example.get("question_type") or "").lower()
    if qtype:
        if qtype in {"multiple-choice", "multi-choice", "multiple_choice"}:
            return True
        if qtype in {"open", "open-ended", "fill-in", "free-form"}:
            return False

    choices = _parse_options(example.get("options"))
    return len(choices) >= 2


def convert_sample_mc(example: dict, idx: int, image_paths: list[str]) -> dict:
    """Convert a multiple-choice MMMU sample to unified format."""
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
        "prompt": "",
    }
    return record


def process_and_save_mc(dataset, jsonl_path: Path, image_dir: Path):
    """Process dataset and save only multiple-choice records to JSONL + image folder."""
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

