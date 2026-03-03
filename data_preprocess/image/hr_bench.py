"""Preprocess HR-Bench 4K dataset to JSONL + image folder format.

Output fields match blink: idx, images, dataset, type, sub_task, question,
choices, answer, prompt. Missing fields (sub_task, prompt) are empty.
Only 4K, multi-choice only, one question per image (cycle_category == 0).
"""

import argparse
import base64
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import datasets
from PIL import Image
from tqdm import tqdm


def load_dataset(input_dir: str):
    parquet_path = Path(input_dir) / "hr_bench_4k.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"hr_bench_4k.parquet not found in {input_dir}")
    ds = datasets.load_dataset("parquet", data_files={"val": str(parquet_path)}, split="val")
    ds = ds.filter(lambda x: x["cycle_category"] == 0 or x["cycle_category"] == "0")
    print(f"Total: {len(ds)} samples (one per image)")
    return ds


def extract_images(example: dict) -> list:
    raw = example.get("image")
    if not raw:
        return []
    if isinstance(raw, bytes):
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    else:
        img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
    return [img]


def _parse_answer(answer_raw: str, choices: list) -> str:
    if not answer_raw:
        return ""
    letter = str(answer_raw).strip("()").upper()
    i = ord(letter) - ord("A") if letter.isalpha() and len(letter) == 1 else -1
    return choices[i] if 0 <= i < len(choices) else answer_raw


def convert_sample(example: dict, idx: int, image_paths: list) -> dict:
    choices = [example.get("A") or "", example.get("B") or "", example.get("C") or "", example.get("D") or ""]
    choices = [c for c in choices if c]
    return {
        "idx": idx,
        "images": image_paths,
        "dataset": "HR-Bench",
        "type": "multi-choice",
        "sub_task": "",
        "question": example.get("question") or "",
        "choices": choices,
        "answer": _parse_answer(example.get("answer") or "", choices),
        "prompt": "",
    }


def save_images(images: list, idx: int, image_dir: Path) -> list:
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
    example, idx, image_dir = args
    image_dir = Path(image_dir)
    images = extract_images(example)
    image_paths = save_images(images, idx, image_dir)
    return convert_sample(example, idx, image_paths)


def process_and_save(dataset, jsonl_path: Path, image_dir: Path, num_proc: int = 8):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(dict(dataset[i]), i, str(image_dir)) for i in range(len(dataset))]
    print(f"Processing with {num_proc} processes...")
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        records = list(tqdm(
            executor.map(process_one_sample, tasks),
            total=len(dataset),
            desc="Processing samples"
        ))
    print("Writing JSONL...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {jsonl_path}")
    print(f"Saved images to {image_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess HR-Bench 4K to JSONL + image folder")
    parser.add_argument("input_dir", type=str, help="Path to HR-Bench dataset directory (containing hr_bench_4k.parquet)")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--image_dir", type=str, required=True, help="Output image folder path")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes")
    args = parser.parse_args()
    dataset = load_dataset(args.input_dir)
    process_and_save(dataset, Path(args.jsonl_path), Path(args.image_dir), args.num_proc)


if __name__ == "__main__":
    main()