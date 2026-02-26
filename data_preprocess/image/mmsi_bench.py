"""Preprocess MMSI-Bench dataset to JSONL + image folder format.

Processing pipeline:
1. load_dataset() - Load source dataset (from HF or local parquet)
2. extract_images() - Extract PIL images from sample
3. convert_sample() - Convert sample to unified format
4. process_and_save() - Coordinate saving images and JSONL

Dataset: RunsenXu/MMSI-Bench (1000 test samples)
Fields: id, images (sequence), question_type, question, answer, thought,
        mean_normed_duration_seconds, difficulty
"""

'''
# Local parquet:
python mmsi_bench.py /mnt/sda/runhaofu/Datasets/MMSI-Bench/ --jsonl_path /mnt/sda/runhaofu/Datasets/test_dataset/mmsi_bench.jsonl --image_dir /mnt/sda/runhaofu/Datasets/test_dataset/mmsi_bench_images --num_proc 8

# From HuggingFace:
python mmsi_bench.py RunsenXu/MMSI-Bench --jsonl_path /path/to/mmsi_bench.jsonl --image_dir /path/to/mmsi_bench_images --num_proc 8
'''
'''
python mmsi_bench.py /mnt/sda/runhaofu/Datasets/MMSI-Bench/ --jsonl_path /mnt/sda/runhaofu/Datasets/test_dataset/mmsi_bench.jsonl --image_dir /mnt/sda/runhaofu/Datasets/test_dataset/mmsi_bench_images --num_proc 8
'''

import argparse
import io
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import datasets
from PIL import Image
from tqdm import tqdm


# ============ Dataset-specific functions (customize per dataset) ============

def load_dataset(input_dir: str):
    """Load MMSI-Bench dataset from local dir (parquet) or HuggingFace.

    Args:
        input_dir: Local path to directory containing MMSI_Bench.parquet,
                   or hub name e.g. "RunsenXu/MMSI-Bench".
    """
    input_path = Path(input_dir)
    parquet_file = input_path / "MMSI_Bench.parquet"

    if parquet_file.exists():
        # Load from local parquet
        ds = datasets.load_dataset(
            "parquet",
            data_files={"test": str(parquet_file)},
            split="test",
        )
        print(f"📂 Loaded from local parquet: {len(ds)} samples")
    else:
        # Load from HuggingFace
        ds = datasets.load_dataset(input_dir, split="test")
        print(f"📂 Loaded from HuggingFace: {len(ds)} samples")

    return ds


def extract_images(example: dict) -> list:
    """Extract PIL images from MMSI-Bench sample.

    MMSI-Bench stores images as list of bytes (binary) in parquet.
    """
    images = example.get("images")
    if images is None:
        return []
    out = []
    for img_data in images:
        if img_data is None:
            continue
        if hasattr(img_data, "save"):
            out.append(img_data)
            continue
        if isinstance(img_data, bytes):
            try:
                out.append(Image.open(io.BytesIO(img_data)).convert("RGB"))
            except Exception:
                continue
        elif isinstance(img_data, dict) and "bytes" in img_data:
            try:
                out.append(
                    Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                )
            except Exception:
                continue
    return out


def _parse_choices_from_question(question: str) -> list[str]:
    """Parse 'Options: A: ..., B: ..., C: ..., D: ...' from question; return choices list (question unchanged)."""
    if not question or "Options:" not in question:
        return []
    parts = question.split("Options:", 1)
    options_block = parts[1].strip() if len(parts) > 1 else ""
    if not options_block:
        return []
    # Content may contain commas; match until next ", X: " (X = A–D) or end
    matches = re.findall(
        r"([A-Z]):\s*(.*?)(?=\s*,\s*[A-Z]:\s*|$)",
        options_block,
        re.DOTALL,
    )
    return [m[1].strip() for m in matches]


def _parse_answer(answer_raw: str, choices: list[str]) -> str:
    """Parse answer letter 'A'/'B'/... -> actual choice text (same as BLINK)."""
    if not answer_raw or not choices:
        return answer_raw
    letter = answer_raw.strip().upper()
    idx = ord(letter) - ord("A") if letter.isalpha() and len(letter) == 1 else -1
    return choices[idx] if 0 <= idx < len(choices) else answer_raw


def convert_sample(example: dict, idx: int, image_paths: list[str]) -> dict:
    """Convert MMSI-Bench sample to unified format (BLINK-style: choices/answer as content)."""
    question = example.get("question", "")
    choices = _parse_choices_from_question(question)
    answer_raw = example.get("answer", "")
    answer_content = _parse_answer(answer_raw, choices) if choices else answer_raw
    record = {
        "idx": idx,
        "images": image_paths,
        "dataset": "MMSI-Bench",
        "type": example.get("question_type", ""),
        "question": question,
        "choices": choices,
        "answer": answer_content,
        "prompt": question,
    }
    return record


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
        description="Preprocess MMSI-Bench to JSONL + image folder"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to MMSI-Bench directory (with MMSI_Bench.parquet) or hub name e.g. RunsenXu/MMSI-Bench",
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
        default=8,
        help="Number of processes for parallel processing",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.input_dir)
    process_and_save(
        dataset, Path(args.jsonl_path), Path(args.image_dir), args.num_proc
    )


if __name__ == "__main__":
    main()
