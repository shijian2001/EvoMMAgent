"""Preprocess MMStar dataset to JSONL + image folder format.

Processing pipeline:
1. load_dataset() - Load source TSV dataset
2. extract_images() - Decode base64 image from sample
3. convert_sample() - Convert sample to unified format
4. process_and_save() - Coordinate saving images and JSONL
"""

'''
python EvoMMAgent/data_preprocess/image/mmstar.py Datasets/MMStar/MMStar.tsv --jsonl_path Datasets/test_dataset/mmstar_val.jsonl --image_dir Datasets/test_dataset/MMStar_images
'''

import argparse
import base64
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import datasets
from PIL import Image
from tqdm import tqdm


# ============ Dataset-specific functions (customize per dataset) ============

def load_dataset(input_dir_or_tsv: str, tsv_filename: str = "MMStar.tsv"):
    """Load MMStar dataset from a directory (preferred) or a local TSV file.

    The TSV is expected to have at least the following columns:
    - index
    - question
    - answer (letter like 'A', 'B', ...)
    - category
    - image (base64-encoded jpeg/png)
    """
    p = Path(input_dir_or_tsv)
    input_path = (p / tsv_filename) if p.is_dir() else p
    print(f"📂 Loading MMStar TSV from {input_path}")
    dataset = datasets.load_dataset(
        "csv",
        data_files={"test": str(input_path)},
        delimiter="\t",
    )["test"]
    print(f"📊 Total: {len(dataset)} samples")
    return dataset


def extract_images(example: dict) -> list:
    """Decode base64 image from MMStar sample and return as a single-item list."""
    img_b64 = example.get("image")
    if not img_b64:
        return []

    try:
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return [img]
    except Exception as e:
        # If decoding fails, skip this sample's image
        print(f"⚠️ Failed to decode image for index {example.get('index')}: {e}")
        return []


def _parse_choices_from_question(question: str) -> list[str]:
    """Parse multiple-choice options from the question text.

    MMStar questions contain an 'Options:' segment, e.g.:
    '...\\nOptions: A: option A text, B: option B text, C: ..., D: ...'
    """
    if not question:
        return []

    # Locate the 'Options:' part (may be on a new line)
    lower = question
    marker = "Options:"
    idx = lower.find(marker)
    if idx == -1:
        return []

    options_str = question[idx + len(marker) :].strip()
    if not options_str:
        return []

    # Split by option letters like 'A:', 'B:', ...
    import re

    parts = re.split(r"\s*[A-Z]:\s*", options_str)
    # First part is text before 'A:', discard it
    raw_choices = parts[1:] if len(parts) > 1 else []

    choices = []
    for c in raw_choices:
        # Remove trailing separators like ',', '.', ';'
        c = c.strip()
        # Cut off anything after a pattern like ', X: ' for safety
        c = re.split(r",\s*[A-Z]:\s*", c)[0].strip()
        c = c.rstrip(".,;").strip()
        if c:
            choices.append(c)

    return choices


def _parse_answer(answer_raw: str, choices: list) -> str:
    """Parse answer letter (e.g. 'A', '(B)') into actual choice text."""
    if answer_raw == "hidden":
        return ""
    if not answer_raw:
        return ""

    letter = answer_raw.strip("()").upper()

    idx = ord(letter) - ord("A") if letter.isalpha() and len(letter) == 1 else -1
    return choices[idx] if 0 <= idx < len(choices) else answer_raw


def convert_sample(example: dict, idx: int, image_paths: list[str]) -> dict:
    """Convert MMStar sample to unified format compatible with BLINK JSONL."""
    question = example.get("question", "")
    choices = _parse_choices_from_question(question)
    answer_text = _parse_answer(example.get("answer", ""), choices)

    return {
        "idx": idx,
        "images": image_paths,
        "dataset": "MMStar",
        "type": "multi-choice",
        # sub_task uses 'category' field from MMStar
        "sub_task": example.get("category", ""),
        "question": question,
        "choices": choices,
        "answer": answer_text,
        # prompt is exactly the question content
        "prompt": question,
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
    """Process one sample: extract images, save them, return record."""
    example, idx, image_dir = args
    images = extract_images(example)
    image_paths = save_images(images, idx, image_dir)
    return convert_sample(example, idx, image_paths)


def process_and_save(dataset, jsonl_path: Path, image_dir: Path, num_proc: int = 8):
    """Process dataset and save to JSONL + image folder."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments
    tasks = [(dataset[i], i, image_dir) for i in range(len(dataset))]

    # Parallel processing
    print(f"🚀 Processing with {num_proc} processes...")
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        records = list(
            tqdm(
                executor.map(process_one_sample, tasks),
                total=len(dataset),
                desc="Processing samples",
            )
        )

    # Write to JSONL
    print("📝 Writing JSONL...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(records)} records to {jsonl_path}")
    print(f"✅ Saved images to {image_dir}")


# ============ Main entry ============

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MMStar to JSONL + image folder"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to MMStar dataset directory (e.g., Datasets/MMStar).",
    )
    parser.add_argument(
        "--tsv_filename",
        type=str,
        default="MMStar.tsv",
        help="TSV filename inside input_dir (default: MMStar.tsv).",
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

    dataset = load_dataset(args.input_dir, tsv_filename=args.tsv_filename)
    process_and_save(dataset, Path(args.jsonl_path), Path(args.image_dir), args.num_proc)


if __name__ == "__main__":
    main()

