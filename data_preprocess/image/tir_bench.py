"""Preprocess TIR-Bench dataset to JSONL + image folder format (same as BLINK).

Only multiple-choice samples (answer is single letter A-Z).
Output fields match blink_val.jsonl: idx, images, dataset, type, sub_task, question, choices, answer, prompt.
Uses same multiprocessing (ProcessPoolExecutor) as blink for speed.
"""

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm


# ============ Dataset-specific functions ============

def _is_multiple_choice_answer(answer) -> bool:
    """Keep only 选择题: answer is single letter A-Z."""
    if not isinstance(answer, str) or len(answer) != 1:
        return False
    return answer.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_dataset(input_dir: str) -> list:
    """Load TIR-Bench.json and filter to multiple-choice only."""
    input_path = Path(input_dir) / "TIR-Bench.json"
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    filtered = [d for d in data if _is_multiple_choice_answer(d.get("answer"))]
    print(f"📂 Loaded {len(data)} samples, kept {len(filtered)} multiple-choice")
    return filtered


def _parse_choices_from_prompt(prompt: str) -> list:
    """Parse choices from prompt lines like 'A: 65%' or 'A. text'."""
    if not prompt:
        return []
    choices = []
    for line in prompt.split("\n"):
        m = re.match(r"^([A-Z])[:\.]\s*(.+)$", line.strip())
        if m:
            choices.append(m.group(2).strip())
    return choices


def extract_images(example: dict, input_dir: Path) -> list:
    """Load PIL images from TIR-Bench image paths (image_1, image_2)."""
    images = []
    for key in ("image_1", "image_2"):
        path = example.get(key)
        if path is None:
            continue
        full_path = input_dir / path
        if not full_path.exists():
            continue
        try:
            img = Image.open(full_path).convert("RGB")
            images.append(img)
        except Exception:
            continue
    return images


def convert_sample(example: dict, idx: int, image_paths: list[str]) -> dict:
    """Convert to same format as blink: all fields present, missing ones empty."""
    prompt_text = example.get("prompt") or ""
    choices = _parse_choices_from_prompt(prompt_text)
    answer_raw = (example.get("answer") or "").strip().upper()
    answer_text = _parse_answer(answer_raw, choices)

    # question: use first line of prompt or full prompt if single line
    lines = prompt_text.strip().split("\n")
    question = lines[0].strip() if lines else ""

    return {
        "idx": idx,
        "images": image_paths,
        "dataset": "TIR-Bench",
        "type": "multi-choice",
        "sub_task": example.get("task") or "",
        "question": question,
        "choices": choices if choices else [],
        "answer": answer_text,
        "prompt": prompt_text,
    }


def _parse_answer(answer_raw: str, choices: list) -> str:
    """Map letter (A, B, ...) to choice text; else return raw."""
    if not answer_raw or len(answer_raw) != 1:
        return answer_raw
    idx = ord(answer_raw.upper()) - ord("A")
    if 0 <= idx < len(choices):
        return choices[idx]
    return answer_raw


# ============ Generic processing (same as blink) ============

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
    """Process one sample: extract images, save them, return record (same pattern as blink)."""
    example, idx, image_dir, input_dir = args
    input_dir = Path(input_dir)
    image_dir = Path(image_dir)
    images = extract_images(example, input_dir)
    if not images:
        # No valid images: still emit record with empty images so idx/fields match
        image_paths = []
    else:
        image_paths = save_images(images, idx, image_dir)
    return convert_sample(example, idx, image_paths)


def process_and_save(dataset: list, jsonl_path: Path, image_dir: Path, input_dir: Path, num_proc: int = 8):
    """Process dataset with multiprocessing and save JSONL + images (same as blink)."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(dataset[i], i, image_dir, input_dir) for i in range(len(dataset))]

    print(f"🚀 Processing with {num_proc} processes...")
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        records = list(
            tqdm(
                executor.map(process_one_sample, tasks),
                total=len(dataset),
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
        description="Preprocess TIR-Bench (multiple-choice only) to JSONL + image folder, same format as BLINK"
    )
    parser.add_argument("input_dir", type=str, help="Path to TIR-Bench dataset directory (containing TIR-Bench.json)")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--image_dir", type=str, required=True, help="Output image folder path")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes for parallel processing")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    dataset = load_dataset(str(input_dir))
    if not dataset:
        print("No multiple-choice samples found.")
        return
    process_and_save(
        dataset,
        Path(args.jsonl_path),
        Path(args.image_dir),
        input_dir,
        args.num_proc,
    )


if __name__ == "__main__":
    main()
