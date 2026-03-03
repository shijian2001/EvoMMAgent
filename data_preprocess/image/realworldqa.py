"""Preprocess RealWorldQA dataset to JSONL + image folder format.

Processing pipeline:
1. load_dataset()   - Load source dataset
2. extract_images() - Extract PIL images from sample
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

def load_dataset(input_dir: str, split: str = "test"):
    """Load RealWorldQA dataset."""
    ds = datasets.load_dataset(input_dir, split=split)
    print(f"📊 Total: {len(ds)} samples")
    return ds


def extract_images(example: dict) -> list:
    """Extract PIL image from RealWorldQA sample."""
    image = example.get("image")
    return [image] if image is not None else []


def _parse_choices_from_question(question: str) -> list[str]:
    """Parse choices from RealWorldQA question text (e.g. 'A. opt1\\nB. opt2\\nC. opt3')."""
    if not question:
        return []
    # Match lines like "A. ..." or "B. ..." (letter + period + space + rest of line)
    pattern = re.compile(r"^([A-D])\.\s*(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(question))
    if not matches:
        return []
    # Sort by letter (A, B, C, D) and return option text only
    ordered = sorted(matches, key=lambda m: m.group(1))
    return [m.group(2).strip() for m in ordered]


def _parse_answer(answer_raw: str, choices: list) -> str:
    """Parse answer format: 'A' or '(A)' -> actual choice text."""
    if not answer_raw:
        return ""
    letter = str(answer_raw).strip("()").upper()
    idx = ord(letter) - ord("A") if letter.isalpha() and len(letter) == 1 else -1
    return choices[idx] if 0 <= idx < len(choices) else str(answer_raw)


def convert_sample(example: dict, idx: int, image_paths: list[str]) -> dict | None:
    """Convert RealWorldQA sample to unified BLINK-style format.

    Fields must match BLINK:
    - idx, images, dataset, type, sub_task, question, choices, answer, prompt
    RealWorldQA has no separate 'choices' column; choices are embedded in question text.
    """
    choices = example.get("choices") or []
    if not isinstance(choices, list) or len(choices) < 2:
        # Parse choices from question (e.g. "A. ...\nB. ...\nC. ...")
        question_raw = example.get("question", "")
        choices = _parse_choices_from_question(question_raw)

    # Only keep multiple-choice questions (at least 2 options)
    if len(choices) < 2:
        return None

    question = example.get("question", "")
    answer = _parse_answer(example.get("answer", ""), choices)

    return {
        "idx": idx,
        "images": image_paths,
        "dataset": "RealWorldQA",
        "type": "multi-choice",
        "sub_task": "",
        "question": question,
        "choices": choices,
        "answer": answer,
        # Use question text as prompt, as requested
        "prompt": question,
    }


# ============ Generic processing functions (reusable across datasets) ============

def save_images(images: list, idx: int, image_dir: Path) -> list[str]:
    """Save images to {idx:05d}/{img_idx:05d}.png and return relative paths."""
    image_paths: list[str] = []
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

    # Parallel processing (same pattern as blink.py)
    print(f"🚀 Processing with {num_proc} processes...")
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        records = list(
            tqdm(
                executor.map(process_one_sample, tasks),
                total=len(dataset),
                desc="Processing samples",
            )
        )

    # Filter out non-multiple-choice samples (convert_sample returned None)
    records = [r for r in records if r is not None]

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
        description="Preprocess RealWorldQA to BLINK-style JSONL + image folder"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path or dataset name for RealWorldQA (e.g., 'xai-org/RealworldQA')",
    )
    parser.add_argument(
        "--jsonl_path", type=str, required=True, help="Output JSONL file path"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Output image folder path"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Number of processes for parallel processing",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.input_dir, split=args.split)
    process_and_save(dataset, Path(args.jsonl_path), Path(args.image_dir), args.num_proc)


if __name__ == "__main__":
    main()

