"""Preprocess BLINK dataset to JSONL + image folder format.

Processing pipeline:
1. load_dataset() - Load source dataset
2. extract_images() - Extract PIL images from sample
3. convert_sample() - Convert sample to unified format
4. process_and_save() - Coordinate saving images and JSONL
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import datasets
from tqdm import tqdm


# ============ Dataset-specific functions (customize per dataset) ============

def load_dataset(input_dir: str):
    """Load BLINK dataset - concatenate all validation configs."""
    configs = datasets.get_dataset_config_names(input_dir)
    print(f"📂 Found {len(configs)} configs")
    
    combined = datasets.concatenate_datasets([
        datasets.load_dataset(input_dir, name=cfg, split="val")
        for cfg in configs
    ])
    print(f"📊 Total: {len(combined)} samples")
    return combined


def extract_images(example: dict) -> list:
    """Extract PIL images from BLINK sample."""
    return [example[f"image_{i}"] for i in range(1, 5) 
            if example[f"image_{i}"] is not None]


def convert_sample(example: dict, idx: int, image_paths: list[str]) -> dict:
    """Convert BLINK sample to unified format."""
    return {
        "idx": idx,
        "images": image_paths,
        "dataset": "BLINK",
        "type": "multi-choice",
        "sub_task": example["sub_task"],
        "question": example["question"],
        "choices": example["choices"],
        "answer": _parse_answer(example["answer"], example["choices"]),
        "prompt": example["prompt"],
    }


def _parse_answer(answer_raw: str, choices: list) -> str:
    """Parse BLINK answer format: '(A)' or 'A' -> actual choice text."""
    if answer_raw == "hidden":
        return ""
    letter = answer_raw.strip("()").upper()
    idx = ord(letter) - ord("A") if letter.isalpha() and len(letter) == 1 else -1
    return choices[idx] if 0 <= idx < len(choices) else answer_raw


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
        records = list(tqdm(
            executor.map(process_one_sample, tasks),
            total=len(dataset),
            desc="Processing samples"
        ))
    
    # Write to JSONL
    print("📝 Writing JSONL...")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(records)} records to {jsonl_path}")
    print(f"✅ Saved images to {image_dir}")


# ============ Main entry ============

def main():
    parser = argparse.ArgumentParser(description="Preprocess BLINK to JSONL + image folder")
    parser.add_argument("input_dir", type=str, help="Path to BLINK dataset directory")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--image_dir", type=str, required=True, help="Output image folder path")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes for parallel processing")
    args = parser.parse_args()
    
    dataset = load_dataset(args.input_dir)
    process_and_save(dataset, Path(args.jsonl_path), Path(args.image_dir), args.num_proc)


if __name__ == "__main__":
    main()
