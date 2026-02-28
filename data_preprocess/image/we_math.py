"""Preprocess We-Math dataset to JSONL + image folder format.

Processing pipeline:
1. load_dataset() - Load source dataset (HuggingFace We-Math/We-Math, testmini split)
2. extract_images() - Extract PIL image from sample
3. convert_sample() - Convert sample to unified format
4. process_and_save() - Coordinate saving images and JSONL
"""

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import datasets
from tqdm import tqdm


# ============ Dataset-specific functions (customize per dataset) ============

def load_dataset(input_dir: str, split: str = "testmini"):
    """Load We-Math dataset from HuggingFace or local path."""
    try:
        ds = datasets.load_dataset(input_dir, split=split, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load We-Math (split={split}). "
            f"Use hub name 'We-Math/We-Math' or local path. Error: {e}"
        ) from e
    print(f"📊 Total: {len(ds)} samples (split={split})")
    return ds


def extract_images(example: dict) -> list:
    """Extract PIL image(s) from We-Math sample."""
    img = example.get("image_path")
    if img is None:
        return []
    if hasattr(img, "save"):
        return [img]
    return []


def _parse_options(option_str: str) -> list:
    """Parse We-Math option string 'A. opt1;B. opt2;...' -> list of choice texts."""
    if not option_str or not isinstance(option_str, str):
        return []
    parts = [p.strip() for p in option_str.split(";") if p.strip()]
    choices = []
    for p in parts:
        m = re.match(r"^[A-Za-z]\.\s*(.*)$", p)
        if m:
            choices.append(m.group(1).strip())
        else:
            choices.append(p)
    return choices


def _parse_answer(answer_raw: str, choices: list) -> str:
    """Parse We-Math answer: letter 'A'/'B'/... -> actual choice text or keep letter."""
    if not answer_raw:
        return ""
    letter = answer_raw.strip().upper()
    idx = ord(letter) - ord("A") if letter.isalpha() and len(letter) == 1 else -1
    return choices[idx] if 0 <= idx < len(choices) else answer_raw


def convert_sample(example: dict, idx: int, image_paths: list[str]) -> dict:
    """Convert We-Math sample to unified format."""
    option_str = example.get("option", "")
    choices = _parse_options(option_str)
    answer_letter = example.get("answer", "")
    return {
        "idx": idx,
        "images": image_paths,
        "dataset": "We-Math",
        "type": "multi-choice",
        "sub_task": example.get("knowledge concept", ""),
        "question": example.get("question", ""),
        "choices": choices,
        "answer": _parse_answer(answer_letter, choices),
        "prompt": "",
    }


# ============ Generic processing functions (reusable across datasets) ============

def save_images(images: list, idx: int, image_dir: Path) -> list[str]:
    """Save images to {idx:05d}/{img_idx:05d}.png and return relative paths."""
    image_paths = []
    sample_dir = image_dir / f"{idx:05d}"
    sample_dir.mkdir(exist_ok=True)

    for img_idx, img in enumerate(images):
        rel_path = f"{idx:05d}/{img_idx:05d}.png"
        full_path = image_dir / rel_path
        img.save(full_path)
        image_paths.append(rel_path)

    return image_paths


def process_one_sample(args):
    """Process one sample: extract images, save them, return record or None if no image."""
    example, idx, image_dir = args
    images = extract_images(example)
    if not images:
        return None
    image_paths = save_images(images, idx, image_dir)
    return convert_sample(example, idx, image_paths)


def process_and_save(dataset, jsonl_path: Path, image_dir: Path, num_proc: int = 8):
    """Process dataset and save to JSONL + image folder."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(dataset[i], i, image_dir) for i in range(len(dataset))]

    print(f"🚀 Processing with {num_proc} processes...")
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        records = list(
            tqdm(
                executor.map(process_one_sample, tasks),
                total=len(dataset),
                desc="Processing samples",
            )
        )

    records = [r for r in records if r is not None]
    if len(records) < len(dataset):
        print(f"⚠️ Skipped {len(dataset) - len(records)} samples (missing image)")

    print("📝 Writing JSONL...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(records)} records to {jsonl_path}")
    print(f"✅ Saved images to {image_dir}")


# ============ Main entry ============

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess We-Math to JSONL + image folder"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        default="We-Math/We-Math",
        nargs="?",
        help="HuggingFace dataset name (e.g. We-Math/We-Math) or local path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="testmini",
        help="Dataset split to load (default: testmini)",
    )
    parser.add_argument(
        "--jsonl_path",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Output image folder path",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=64,
        help="Number of processes for parallel processing",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.input_dir, split=args.split)
    process_and_save(
        dataset, Path(args.jsonl_path), Path(args.image_dir), args.num_proc
    )


if __name__ == "__main__":
    main()
