"""Preprocess MathVerse dataset to BLINK-val-compatible multi-choice JSONL + image folder.
"""
'''
python EvoMMAgent/data_preprocess/image/mathverse_mc.py /mnt/sda/runhaofu/Datasets/MathVerse --jsonl_path /mnt/sda/runhaofu/Datasets/test_dataset/mathverse_mc.jsonl --image_dir /mnt/sda/runhaofu/Datasets/test_dataset/mathverse_mc_images --num_proc 8
'''


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


def load_dataset(input_dir: str):
    """Load MathVerse dataset - testmini split (with images)."""
    try:
        ds = datasets.load_dataset(input_dir, name="testmini", split="testmini")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MathVerse testmini. "
            f"Use local path or 'AI4Math/MathVerse'. Error: {e}"
        ) from e
    print(f"📊 Total: {len(ds)} samples (testmini)")
    return ds


def extract_images(example: dict) -> list:
    """Extract PIL image(s) from MathVerse sample (single image per sample)."""
    img = example.get("image")
    if img is None:
        return []
    return [img] if hasattr(img, "save") else []


def _question_has_options(question: str) -> bool:
    if not question:
        return False
    return bool(_RE_HAS_CHOICES_BLOCK.search(question) or _RE_HAS_OPTION_LINES.search(question))


def _parse_choices_from_question(question: str) -> list[str]:
    """Parse 'Choices:\\nA:...\\nB:...' block from question."""
    if not question or "Choices:" not in question:
        return []
    parts = question.split("Choices:", 1)
    choices_block = parts[1].strip() if len(parts) > 1 else ""
    choices: list[str] = []
    for line in choices_block.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([A-Z])[:\s\)]\s*(.*)$", line, re.IGNORECASE)
        if m:
            choices.append(m.group(2).strip())
        else:
            choices.append(line)
    return choices


def _parse_answer(answer_raw: str, choices: list[str]) -> str:
    """Match BLINK behavior: '(A)'/'A' -> choice text; 'hidden' -> ''."""
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
    """Convert MathVerse sample to BLINK-val-compatible record; return None if filtered out."""
    if example.get("question_type") != "multi-choice":
        return None

    metadata = example.get("metadata") or {}
    if hasattr(metadata, "keys"):
        metadata = {k: metadata[k] for k in metadata.keys()}

    question = example.get("question", "") or ""
    # 只保留题干非空的选择题（Vision Only 的 question=="" 直接丢弃）
    if not question.strip():
        return None

    # choices 从 question 解析；若 question 为空（Vision Only），允许从 question_for_eval 解析 choices
    choices = _parse_choices_from_question(question)
    if not choices:
        choices = _parse_choices_from_question(example.get("question_for_eval", "") or "")

    # 只保留“真正有选项”的选择题
    if len(choices) < 2:
        return None

    answer = _parse_answer(example.get("answer", ""), choices)
    prompt = question if _question_has_options(question) else ""

    sub_task = ""
    sub_like = metadata.get("subfield", "")
    if sub_like is not None:
        sub_task = str(sub_like) if str(sub_like).strip() else ""

    record = {
        "idx": idx,
        "images": image_paths,
        "dataset": "MathVerse",
        "type": "multi-choice",
        "sub_task": sub_task,
        "question": question,
        "choices": choices,
        "answer": answer,
        "prompt": prompt,
    }
    return {k: record.get(k, "") for k in BLINK_FIELDS}


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
    """Process one sample: extract images, save them, return record (or None if filtered)."""
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

    # filter Nones
    records = [r for r in records if r is not None]

    print("📝 Writing JSONL...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(records)} records to {jsonl_path}")
    print(f"✅ Saved images to {image_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MathVerse (testmini) to BLINK-val-compatible multi-choice JSONL + image folder"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to MathVerse dataset dir or hub name (e.g. AI4Math/MathVerse)",
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

