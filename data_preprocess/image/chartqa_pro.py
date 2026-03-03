"""Preprocess ChartQAPro dataset to JSONL + image folder format (multi-choice only)."""

import argparse
import io
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import datasets
from PIL import Image
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

_RE_OPTION_LINES = re.compile(r"(?m)^\s*([A-D])[\.\)\:]?\s*(.+?)(?=\s*(?:[A-D][\.\)\:]|\s*$))", re.DOTALL)


def load_dataset(input_dir: str, split: str = "test"):
    try:
        ds = datasets.load_dataset(input_dir, split=split)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load ChartQAPro split='{split}'. "
            f"Use 'ahmed-masry/ChartQAPro' or local path. Error: {e}"
        ) from e
    qt_key = "Question Type" if "Question Type" in ds.column_names else "QuestionType"
    mcq = ds.filter(lambda x: (x.get(qt_key) or "").strip().upper() == "MULTI CHOICE")
    
    total = len(mcq)
    unanswerable = sum(
        1 for x in mcq 
        if str(x.get("Answer", "")).strip().lower() in ["unanswerable", "n/a", "none"]
    )
    
    print(f"Total: {len(ds)} samples, Multi Choice: {total}")
    print(f"Unanswerable answers: {unanswerable} ({100*unanswerable/total:.1f}%)")
    
    return mcq


def extract_images(example: dict) -> list:
    img = example.get("image") or example.get("Image")
    if img is None:
        return []
    if hasattr(img, "save"):
        return [img]
    if isinstance(img, bytes):
        return [Image.open(io.BytesIO(img)).convert("RGB")]
    return []


def _normalize_sequence(val) -> list:
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [str(x).strip() for x in val if str(x).strip()]
    return [str(val).strip()] if str(val).strip() else []


def _parse_choices_from_question(question: str) -> list:
    if not question or not question.strip():
        return []
    choices = []
    for m in _RE_OPTION_LINES.finditer(question):
        choices.append(m.group(2).strip())
    if not choices:
        for line in question.split("\n"):
            m = re.match(r"^\s*([A-D])[\.\)\:]?\s*(.+)$", line.strip(), re.IGNORECASE)
            if m:
                choices.append(m.group(2).strip())
    return choices


def _answer_text(answer_seq: list, choices: list) -> str:
    if not answer_seq:
        return ""
    a = answer_seq[0].strip()
    if not a:
        return ""
    letter = a.strip("()").upper()
    if letter.isalpha() and len(letter) == 1:
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]
    return a


def convert_sample(example: dict, idx: int, image_paths: list[str]) -> dict | None:
    question_seq = _normalize_sequence(example.get("Question"))
    question = question_seq[0] if question_seq else (example.get("Question") or "")
    if isinstance(question, list):
        question = question[0] if question else ""
    question = str(question).strip()
    if not question:
        return None

    choices = _parse_choices_from_question(question)
    answer_seq = _normalize_sequence(example.get("Answer"))
    answer = _answer_text(answer_seq, choices)
    if not answer and answer_seq:
        answer = answer_seq[0]

    if not answer or answer.lower() in ["unanswerable", "n/a", "none"]:
        return None
    
    if choices and answer not in choices:
        return None

    record = {
        "idx": idx,
        "images": image_paths,
        "dataset": "ChartQAPro",
        "type": "multi-choice",
        "sub_task": "",
        "question": question,
        "choices": choices,
        "answer": answer,
        "prompt": question,
    }
    return {k: record.get(k, "" if k != "choices" else []) for k in BLINK_FIELDS}


def save_images(images: list, idx: int, image_dir: Path) -> list[str]:
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
    images = extract_images(example)
    if not images:
        return None
    image_paths = save_images(images, idx, image_dir)
    return convert_sample(example, idx, image_paths)


def process_and_save(dataset, jsonl_path: Path, image_dir: Path, num_proc: int = 8):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(dataset[i], i, image_dir) for i in range(len(dataset))]

    print(f"Processing with {num_proc} processes...")
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        records = list(
            tqdm(
                executor.map(process_one_sample, tasks),
                total=len(dataset),
                desc="Processing samples",
            )
        )

    records = [r for r in records if r is not None]
    
    # ✅ 统计过滤结果
    filtered_count = len(dataset) - len(records)
    print(f"\n{'='*60}")
    print(f"Total samples processed: {len(dataset)}")
    print(f"Valid samples: {len(records)}")
    print(f"Filtered out: {filtered_count} ({100*filtered_count/len(dataset):.1f}%)")
    print(f"{'='*60}\n")

    print("Writing JSONL...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} records to {jsonl_path}")
    print(f"Saved images to {image_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ChartQAPro (MCQ only) to BLINK-format JSONL + image folder"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="ChartQAPro dataset path or hub name (e.g. ahmed-masry/ChartQAPro)",
    )
    parser.add_argument("--jsonl_path", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--image_dir", type=str, required=True, help="Output image folder")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes")
    args = parser.parse_args()

    dataset = load_dataset(args.input_dir, split=args.split)
    process_and_save(
        dataset, Path(args.jsonl_path), Path(args.image_dir), args.num_proc
    )


if __name__ == "__main__":
    main()
