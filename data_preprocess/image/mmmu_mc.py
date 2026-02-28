"""Preprocess MMMU dataset to JSONL + image folder format, keeping only multiple-choice questions."""

import argparse
import ast
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import datasets
from tqdm import tqdm


def load_dataset(input_dir: str, split: str = "validation"):
    """Load MMMU dataset and concatenate all subject configs for a given split."""
    try:
        configs = datasets.get_dataset_config_names(input_dir)
    except Exception:
        configs = []
    if not configs:
        ds = datasets.load_dataset(input_dir, split=split)
        print(f"Loaded {input_dir} ({split}): {len(ds)} samples")
        return ds

    print(f"Found {len(configs)} configs, loading split={split}...")
    parts = []
    for cfg in configs:
        try:
            part = datasets.load_dataset(input_dir, name=cfg, split=split)
            parts.append(part)
        except Exception as e:
            print(f"Skip config {cfg}: {e}")
    if not parts:
        raise RuntimeError(f"No data loaded for split={split}")
    combined = datasets.concatenate_datasets(parts)
    print(f"Total: {len(combined)} samples")
    return combined


def extract_images(example: dict) -> list:
    """Extract PIL images from MMMU sample (image_1 .. image_5)."""
    out = []
    for i in range(1, 6):
        key = f"image_{i}"
        img = example.get(key)
        if img is not None and hasattr(img, "save"):
            out.append(img)
    return out


def _parse_options(options) -> list:
    """Parse MMMU options into list of choice strings."""
    if options is None:
        return []
    if isinstance(options, list):
        return [str(x).strip() for x in options if x is not None]
    s = str(options).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x is not None]
        except (ValueError, SyntaxError):
            pass
    parts = re.split(r"\s*[\(]?[A-Za-z][\)]\s*", s)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        return parts
    return [line.strip() for line in s.split("\n") if line.strip()]


def _normalize_answer(answer_raw, choices: list) -> str:
    """Normalize answer label/content to choice text when possible."""
    if answer_raw is None:
        return ""
    if isinstance(answer_raw, list):
        if len(answer_raw) == 1:
            return str(answer_raw[0]).strip()
        return str(answer_raw).strip()
    s = str(answer_raw).strip()
    if not s:
        return ""
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list) and len(parsed) == 1:
                return str(parsed[0]).strip()
            if isinstance(parsed, list) and parsed:
                return str(parsed).strip()
        except (ValueError, SyntaxError):
            pass
    letter = s.strip("()").upper()
    if len(letter) == 1 and letter.isalpha():
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]
    return s


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


def process_one_sample(args):
    """Process one sample: extract images, save them, return record."""
    example, idx, image_dir = args
    images = extract_images(example)
    image_paths = save_images(images, idx, image_dir)
    return convert_sample_mc(example, idx, image_paths)


def process_and_save_mc(dataset, jsonl_path: Path, image_dir: Path, num_proc: int = 8):
    """Process dataset and save only multiple-choice records to JSONL + image folder."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 Filtering multiple-choice samples...")
    mc_samples = [(ex, i) for i, ex in enumerate(tqdm(dataset, desc="Filtering")) 
                  if is_multiple_choice(ex)]
    print(f"📊 Found {len(mc_samples)} MC samples out of {len(dataset)}")

    tasks = [(example, new_idx, image_dir) for new_idx, (example, _) in enumerate(mc_samples)]

    print(f"🚀 Processing with {num_proc} processes...")
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        records = list(tqdm(
            executor.map(process_one_sample, tasks),
            total=len(tasks),
            desc="Processing samples"
        ))

    # Step 4: 写入 JSONL
    print("📝 Writing JSONL...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Total samples in source dataset: {len(dataset)}")
    print(f"✅ Kept multiple-choice samples: {len(records)}")
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
        default="test",
        choices=["dev", "validation", "test"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Number of processes for parallel processing",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.input_dir, split=args.split)
    process_and_save_mc(dataset, Path(args.jsonl_path), Path(args.image_dir), args.num_proc)


if __name__ == "__main__":
    main()