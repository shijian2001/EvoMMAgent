"""Preprocess TreeBench dataset to JSONL + image folder format (same schema as BLINK).

- Only multi-choice questions (rows with at least one of A,B,C,D non-empty).
- Output fields match blink: idx, images, dataset, type, sub_task, question, choices, answer, prompt.
- Uses ProcessPoolExecutor for parallel processing like blink.
"""

import argparse
import base64
import csv
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

# TSV may have very large fields (base64 image)
csv.field_size_limit(10**9)

# Blink-compatible output keys (missing fields → empty string)
BLINK_KEYS = ("idx", "images", "dataset", "type", "sub_task", "question", "choices", "answer", "prompt")

CHOICE_COLS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]


def _iter_tsv_rows(tsv_path: str):
    """Iterate TreeBench TSV rows as dicts (only multi-choice rows)."""
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            return
        out_idx = 0
        for row in reader:
            choices = _get_choices(row)
            if not choices:
                continue
            row["_choices"] = choices
            row["_out_idx"] = out_idx
            out_idx += 1
            yield row


def _get_choices(row: dict) -> list:
    """Build choices list from A,B,C,... columns (order preserved, empty skipped only at end)."""
    out = []
    for c in CHOICE_COLS:
        v = (row.get(c) or "").strip()
        if v:
            out.append(v)
    return out


def _parse_answer(answer_raw: str, choices: list) -> str:
    """Parse answer: 'A' or '(A)' -> corresponding choice text; else return raw."""
    if not (answer_raw or "").strip():
        return ""
    letter = str(answer_raw).strip().strip("()").upper()
    if len(letter) == 1 and letter.isalpha():
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]
    return answer_raw.strip()


def process_one_sample(args):
    """Decode image, save to image_dir, return one record in blink format."""
    row, image_dir_str = args
    idx = row["_out_idx"]
    choices = row["_choices"]
    image_dir = Path(image_dir_str)

    # Decode base64 image and save (single image per sample)
    image_paths = []
    b64 = (row.get("image") or "").strip()
    if b64:
        try:
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            sample_dir = image_dir / f"{idx:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            rel = f"{idx:05d}/00000.png"
            img.save(image_dir / rel)
            image_paths.append(rel)
        except Exception:
            pass

    if not image_paths:
        image_paths = [f"{idx:05d}/00000.png"]

    answer = _parse_answer(row.get("answer") or "", choices)
    sub_task = (row.get("category") or "").strip()
    question = (row.get("question") or "").strip()

    record = {
        "idx": idx,
        "images": image_paths,
        "dataset": "TreeBench",
        "type": "multi-choice",
        "sub_task": sub_task,
        "question": question,
        "choices": choices,
        "answer": answer,
        "prompt": "",
    }
    return record


def _task_iter(tsv_path: str, image_dir_str: str, max_samples: Optional[int] = None):
    """Generator of (row, image_dir_str) for multi-choice rows (for streaming to pool)."""
    n = 0
    for row in _iter_tsv_rows(tsv_path):
        yield (row, image_dir_str)
        n += 1
        if max_samples is not None and n >= max_samples:
            break


TSV_FILENAME = "TreeBench.tsv"


def run(input_dir: str, jsonl_path: str, image_dir: str, num_proc: int = 8, max_samples: Optional[int] = None):
    input_dir = Path(input_dir)
    tsv_path = input_dir / TSV_FILENAME
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Not found: {tsv_path} (input_dir should be the folder containing TreeBench.tsv)")
    jsonl_path = Path(jsonl_path)
    image_dir = Path(image_dir)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    tasks = _task_iter(str(tsv_path), str(image_dir), max_samples)
    print(f"🚀 Processing with {num_proc} processes (multi-choice only)...")
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        records = list(
            tqdm(
                executor.map(process_one_sample, tasks, chunksize=1),
                desc="Processing samples",
            )
        )
    if not records:
        print("No multi-choice samples found.")
        return

    print("📝 Writing JSONL...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(records)} records to {jsonl_path}")
    print(f"✅ Saved images to {image_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess TreeBench TSV to JSONL + image folder (blink schema)"
    )
    parser.add_argument("input_dir", type=str, help="Path to folder containing TreeBench.tsv")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--image_dir", type=str, required=True, help="Output image folder")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process (for testing)")
    args = parser.parse_args()
    run(args.input_dir, args.jsonl_path, args.image_dir, args.num_proc, args.max_samples)


if __name__ == "__main__":
    main()
