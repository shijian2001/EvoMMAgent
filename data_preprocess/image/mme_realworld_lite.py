"""Preprocess MME-RealWorld-lite-lmms-eval to JSONL + image folder format.

Processing pipeline:
1. load_dataset() - Load source dataset
2. extract_images() - Extract PIL images from sample
3. convert_sample() - Convert sample to unified format (BLINK-style fields)
4. process_and_save() - Coordinate saving images and JSONL with multiprocessing
"""

import argparse
import base64
import json
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path
import os

import datasets
from PIL import Image
from tqdm import tqdm


# ============ Dataset-specific functions ============

def load_dataset(input_dir: str, split: str = "train"):
    """Load MME-RealWorld-lite-lmms-eval dataset (defaults to train split)."""
    try:
        ds = datasets.load_dataset(input_dir, split=split)
        used_split = split
    except Exception:
        try:
            ds = datasets.load_dataset(input_dir, split="train")
            used_split = "train"
        except Exception:
            ds = datasets.load_from_disk(input_dir)
            used_split = "from_disk"
    print(f"📊 Loaded {len(ds)} samples from split '{used_split}'")
    return ds


def _ensure_pil_image(example: dict):
    """Return PIL.Image from common field patterns."""
    if "image" in example:
        return example["image"]
    # Prefer bytes/base64 field when available (Juhe/HF export often stores image here)
    if "bytes" in example:
        data = example["bytes"]
        if isinstance(data, (bytes, bytearray)):
            return Image.open(BytesIO(data))
        if isinstance(data, str):
            # Try base64-decoding string bytes
            try:
                raw = base64.b64decode(data)
                return Image.open(BytesIO(raw))
            except Exception as e:
                raise TypeError("Failed to decode base64 string in 'bytes' field.") from e
    # Fall back to explicit file path only if it actually exists (some paths are dummy)
    if "path" in example:
        p = example["path"]
        if isinstance(p, str) and os.path.exists(p):
            return Image.open(p)
    raise KeyError("No usable image field found in example (expected 'image', 'bytes' or existing 'path').")


def extract_images(example: dict) -> list:
    """Extract single image from MME-RealWorld sample."""
    return [_ensure_pil_image(example)]


def _parse_answer(answer_raw: str, raw_choices: list) -> str:
    """Parse answer to clean choice text.

    Supports:
    - Letter formats like 'A', '(B)', 'C)' -> map to corresponding choice.
    - Full option string equal to one of the raw choices '(A) xxx' -> map to cleaned version.
    """
    if not answer_raw:
        return ""

    # Case 1: answer is exactly one of the raw choice strings
    for c in raw_choices:
        if answer_raw == c:
            return _clean_choice(c)

    # Case 2: answer is a letter (with optional brackets)
    letter = answer_raw.strip()
    if letter.startswith("("):
        letter = letter[1:]
    if letter.endswith(")"):
        letter = letter[:-1]
    letter = letter.strip().upper()
    if not letter:
        return answer_raw
    idx = ord(letter[0]) - ord("A") if letter[0].isalpha() else -1
    cleaned_choices = [_clean_choice(c) for c in raw_choices]
    return cleaned_choices[idx] if 0 <= idx < len(cleaned_choices) else answer_raw


def _clean_choice(text: str) -> str:
    """Remove leading option labels like 'A. ', 'B) ' from a choice string."""
    if not isinstance(text, str):
        return text
    stripped = text.lstrip()
    # Patterns like "(A) xxx"
    if len(stripped) >= 4 and stripped[0] == "(" and stripped[2] == ")" and stripped[1].isalpha():
        return stripped[3:].lstrip()
    # Patterns like "A. xxx", "B) xxx", "C: xxx"
    if len(stripped) >= 3 and stripped[0].isalpha() and stripped[1] in [".", ")", "：", ":"]:
        return stripped[2:].lstrip()
    return stripped


def convert_sample(example: dict, idx: int, image_paths: list[str]) -> dict:
    """Convert MME-RealWorld sample to unified BLINK-style format."""
    raw_choices = example.get("multi-choice options", []) or []
    choices = [_clean_choice(c) for c in raw_choices]
    return {
        "idx": idx,
        "images": image_paths,
        "dataset": "MME-RealWorld-Lite",
        "type": "multi-choice",
        "sub_task": example.get("category", ""),
        "question": example.get("question", ""),
        "choices": choices,
        "answer": _parse_answer(example.get("answer", ""), raw_choices),
        "prompt": "",
    }


# ============ Generic processing functions ============

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
    """Process one sample: extract images, save them, return record."""
    example, global_idx, image_dir = args
    images = extract_images(example)
    image_paths = save_images(images, global_idx, image_dir)
    return convert_sample(example, global_idx, image_paths)


def process_and_save(dataset, jsonl_path: Path, image_dir: Path, num_proc: int = 8):
    """Process dataset and save to JSONL + image folder (multi-choice only)."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Collect only multi-choice examples, then assign new sequential idx
    filtered = []
    for ex in dataset:
        options = ex.get("multi-choice options", []) or []
        if options:
            filtered.append(ex)

    tasks = [(ex, idx, image_dir) for idx, ex in enumerate(filtered)]

    print(f"✅ Collected {len(tasks)} multi-choice samples")

    print(f"🚀 Processing with {num_proc} processes...")
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        records = list(
            tqdm(
                executor.map(process_one_sample, tasks),
                total=len(tasks),
                desc="Processing samples",
            )
        )

    print("📝 Writing JSONL...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(records)} records to {jsonl_path}")
    print(f"✅ Saved images to {image_dir}")


# ============ Main entry ============

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MME-RealWorld-lite-lmms-eval to BLINK-style JSONL + image folder"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="HuggingFace dataset path or local directory for MME-RealWorld-lite-lmms-eval",
    )
    parser.add_argument(
        "--jsonl_path", type=str, required=True, help="Output JSONL file path"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Output image folder path"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Number of processes for parallel processing",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.input_dir)
    process_and_save(dataset, Path(args.jsonl_path), Path(args.image_dir), args.num_proc)


if __name__ == "__main__":
    main()

