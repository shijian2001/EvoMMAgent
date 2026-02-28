"""Preprocess MathVision dataset to JSONL + image folder format, keeping only multi-choice questions.

Processing pipeline:
1. load_dataset() - Load source dataset
2. extract_images() - Extract PIL images from sample
3. convert_sample() - Convert sample to unified format (filters non-MC)
4. process_and_save() - Coordinate saving images and JSONL
"""

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import datasets
from tqdm import tqdm


BLINK_FIELDS = (
    "idx",
    "images",
    "dataset",
    "type",
    "sub_task",
    "question",
    "choices",
    "answer",
    "prompt",
)

_RE_HAS_CHOICES_BLOCK = re.compile(r"(?is)\bchoices\s*:\s*\n")
_RE_HAS_OPTION_LINES = re.compile(r"(?m)^\s*([A-D])\s*[:\)]\s*.+$")


def load_dataset(input_dir: str, split: str):
    """Load MathVision split from local path or hub name (e.g. 'MathLLMs/MathVision')."""
    try:
        ds = datasets.load_dataset(input_dir, split=split)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MathVision split='{split}'. "
            f"Use local path or 'MathLLMs/MathVision'. Error: {e}"
        ) from e
    print(f"📊 Total: {len(ds)} samples ({split})")
    return ds


def extract_images(example: dict) -> list:
    """Extract PIL image(s) from MathVision sample."""
    img = example.get("decoded_image") or example.get("image")
    if img is None:
        return []
    return [img] if hasattr(img, "save") else []


def _question_has_options(question: str) -> bool:
    if not question:
        return False
    return bool(_RE_HAS_CHOICES_BLOCK.search(question) or _RE_HAS_OPTION_LINES.search(question))


def _parse_choices_from_question(question: str) -> list[str]:
    """Parse option lines like 'A: ...' / 'B) ...' from question text."""
    if not question:
        return []
    choices: list[str] = []
    for line in question.split("\n"):
        m = re.match(r"^\s*([A-D])\s*[:\)]\s*(.+?)\s*$", line.strip(), re.IGNORECASE)
        if m:
            choices.append(m.group(2).strip())
    return choices


def _normalize_options(options_raw) -> list[str]:
    if options_raw is None:
        return []
    if isinstance(options_raw, (list, tuple)):
        return [str(x).strip() for x in options_raw if str(x).strip()]
    return []


def _parse_answer(answer_raw: str, choices: list[str]) -> str:
    """Parse answer: '(A)'/'A' -> choice text; 'hidden' -> ''."""
    if answer_raw is None:
        return ""
    a = str(answer_raw).strip()
    if a == "hidden":
        return ""
    if not choices:
        return a
    letter = a.strip("()").strip().upper()
    idx = ord(letter) - ord("A") if letter.isalpha() and len(letter) == 1 else -1
    return choices[idx] if 0 <= idx < len(choices) else a


def convert_sample(example: dict, idx: int, image_paths: list[str]) -> dict | None:
    """Convert MathVision sample to unified format; return None if not multi-choice."""
    question = example.get("question", "") or ""

    choices = _normalize_options(example.get("options"))
    if len(choices) < 2:
        choices = _parse_choices_from_question(question)
    if len(choices) < 2:
        return None

    answer = _parse_answer(example.get("answer", ""), choices)

    sub_task_raw = example.get("subject", "")
    sub_task = str(sub_task_raw).strip() if sub_task_raw is not None else ""

    prompt = ""

    record = {
        "idx": idx,
        "images": image_paths,
        "dataset": "MathVision",
        "type": "multi-choice",
        "sub_task": sub_task,
        "question": question,
        "choices": choices,
        "answer": answer,
        "prompt": prompt,
    }
    return {k: record.get(k, "") for k in BLINK_FIELDS}


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
    for new_idx, rec in enumerate(records):
        rec["idx"] = new_idx

    print("📝 Writing JSONL...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(records)} records to {jsonl_path}")
    print(f"✅ Saved images to {image_dir}")


# ============ Main entry ============

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MathVision to multi-choice JSONL + image folder"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to MathVision dataset dir or hub name (e.g. MathLLMs/MathVision)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "testmini"],
        help="Which split to process (default: full test set)",
    )
    parser.add_argument("--jsonl_path", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--image_dir", type=str, required=True, help="Output image folder path")
    parser.add_argument(
        "--num_proc",
        type=int,
        default=64,
        help="Number of processes for parallel processing",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.input_dir, args.split)
    process_and_save(dataset, Path(args.jsonl_path), Path(args.image_dir), args.num_proc)


if __name__ == "__main__":
    main()
