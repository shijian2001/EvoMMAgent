"""Preprocess vstar_bench dataset to JSONL + image folder format.

Processing pipeline:
1. load_dataset() - Load test_questions.jsonl from input dir
2. resolve_image_path() - Get full path to image file
3. convert_sample() - Convert sample to unified format
4. process_and_save() - Copy images and write JSONL
"""

'''
python vstar_bench.py /mnt/sda/runhaofu/Datasets/vstar_bench/ --jsonl_path /mnt/sda/runhaofu/Datasets/test_dataset/vstar_bench_val.json --image_dir /mnt/sda/runhaofu/Datasets/test_dataset/vstar_bench_images --num_proc 8
'''
import argparse
import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm


# ============ Dataset-specific functions (customize per dataset) ============

def load_dataset(input_dir: str) -> list:
    """Load vstar_bench dataset from test_questions.jsonl."""
    input_path = Path(input_dir) / "test_questions.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected {input_path}")
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"📂 Loaded {len(records)} samples from {input_path}")
    return records


def resolve_image_path(input_dir: Path, image_rel: str) -> Path:
    """Resolve image path: try input_dir/image_rel then .cache/.../image_rel."""
    direct = input_dir / image_rel
    if direct.exists():
        return direct
    cache_path = input_dir / ".cache" / "huggingface" / "download" / image_rel
    if cache_path.exists():
        return cache_path
    return direct  # caller will get FileNotFoundError if missing


def parse_text_to_question_choices(text: str) -> tuple:
    """Parse vstar_bench text into question and choices list.
    Format: 'Question?\n(A) opt1\n(B) opt2\n...\nAnswer with...'
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    question_parts = []
    choices = []
    choice_pat = re.compile(r"^\(([A-Z])\)\s*(.*)$")
    for line in lines:
        m = choice_pat.match(line)
        if m:
            choices.append(m.group(2).strip())
        else:
            if "Answer with the option" in line or "answer with" in line.lower():
                break
            question_parts.append(line)
    question = " ".join(question_parts).strip() if question_parts else text.split("\n")[0]
    return question, choices


def _parse_answer(label: str, choices: list) -> str:
    """Parse label 'A'/'B'/... to actual choice text."""
    if not label or not choices:
        return label or ""
    letter = label.strip().upper()
    idx = ord(letter) - ord("A") if letter.isalpha() and len(letter) == 1 else -1
    return choices[idx] if 0 <= idx < len(choices) else label


def convert_sample(example: dict, idx: int, image_paths: list) -> dict:
    """Convert vstar_bench sample to unified format (BLINK-style)."""
    text = example.get("text", "")
    # Only use parsed choices; keep question identical to prompt text
    _, choices = parse_text_to_question_choices(text)
    return {
        "idx": idx,
        "images": image_paths,
        "dataset": "vstar_bench",
        "type": "multi-choice",
        "sub_task": example.get("category", ""),
        "question": text,
        "choices": choices,
        "answer": _parse_answer(example.get("label", ""), choices),
        "prompt": text,
    }


# ============ Generic processing functions (reusable across datasets) ============

def copy_image_to_sample_dir(src_path: Path, idx: int, image_dir: Path) -> str:
    """Copy one image to {image_dir}/{idx:05d}/00000.png, return relative path."""
    rel_path = f"{idx:05d}/00000.png"
    dest_dir = image_dir / f"{idx:05d}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = image_dir / rel_path
    shutil.copy2(src_path, dest_file)
    return rel_path


def process_one_sample(args):
    """Process one sample: resolve image, copy, return record or None on error."""
    example, idx, input_dir, image_dir = args
    input_path = Path(input_dir)
    image_rel = example.get("image", "")
    if not image_rel:
        return None
    try:
        src = resolve_image_path(input_path, image_rel)
        if not src.exists():
            return None
        image_paths = [copy_image_to_sample_dir(src, idx, Path(image_dir))]
        return convert_sample(example, idx, image_paths)
    except Exception:
        return None


def process_and_save(dataset: list, jsonl_path: Path, image_dir: Path, input_dir: str, num_proc: int = 8):
    """Process dataset and save to JSONL + image folder."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(dataset[i], i, input_dir, str(image_dir)) for i in range(len(dataset))]

    print(f"🚀 Processing with {num_proc} processes...")
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        records = list(
            tqdm(
                executor.map(process_one_sample, tasks),
                total=len(dataset),
                desc="Processing samples",
            )
        )

    # Drop failed
    records = [r for r in records if r is not None]
    if len(records) < len(dataset):
        print(f"⚠️ Skipped {len(dataset) - len(records)} samples (missing image or error)")

    print("📝 Writing JSONL...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(records)} records to {jsonl_path}")
    print(f"✅ Saved images to {image_dir}")


# ============ Main entry ============

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess vstar_bench to JSONL + image folder"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to vstar_bench dataset directory (containing test_questions.jsonl and category image folders)",
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
        dataset,
        Path(args.jsonl_path),
        Path(args.image_dir),
        args.input_dir,
        args.num_proc,
    )


if __name__ == "__main__":
    main()
