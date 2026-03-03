"""Preprocess ZoomBench dataset to JSONL + image folder format (BLINK schema).
MCQ samples only. Output fields: idx, images, dataset, type, sub_task, question, choices, answer, prompt.
"""

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import datasets
from PIL import Image
from tqdm import tqdm


def _parse_options(options_str) -> list:
    if not options_str:
        return []
    if isinstance(options_str, list):
        return [str(x).strip() for x in options_str if x]
    s = str(options_str).strip()
    if not s:
        return []
    pattern = re.compile(
        r"\(?([A-F])\)?\s*[.:]\s*(.*?)(?=\s*\(?[A-F]\)?\s*[.:]\s*|\s*$)",
        re.DOTALL | re.IGNORECASE,
    )
    matches = pattern.findall(s)
    if matches:
        return [m[1].strip() for m in matches]
    return []


def _choices_from_question(question: str) -> list:
    choices = []
    for line in question.split("\n"):
        line = line.strip()
        m = re.match(r"^[A-F][.)]\s*(.*)$", line, re.IGNORECASE)
        if m:
            choices.append(m.group(1).strip())
    return choices


def _parse_answer(answer_raw: str, choices: list) -> str:
    if not answer_raw or not isinstance(answer_raw, str):
        return (answer_raw or "").strip()
    letter = answer_raw.strip("()").upper()
    if len(letter) != 1 or not letter.isalpha():
        return answer_raw.strip()
    idx = ord(letter) - ord("A")
    return choices[idx] if 0 <= idx < len(choices) else answer_raw.strip()


def load_dataset(input_dir: str, quiet: bool = False):
    input_path = Path(input_dir)
    if input_path.is_dir() and (input_path / "dataset_infos.json").exists():
        ds = datasets.load_dataset(str(input_path), split="test")
    else:
        ds = datasets.load_dataset(input_dir, split="test")
    mcq = ds.filter(
        lambda qt: (qt or "").lower() == "mcq",
        input_columns=["question_type"],
    )
    if not quiet:
        print(f"Total: {len(ds)}, MCQ: {len(mcq)}")
    return mcq


def extract_images(example: dict) -> list:
    out = []
    img = example.get("image")
    if img is not None:
        if hasattr(img, "convert"):
            out.append(img.convert("RGB") if img.mode != "RGB" else img)
        else:
            try:
                out.append(Image.open(img).convert("RGB"))
            except Exception:
                pass
    return out


def convert_sample(example: dict, idx: int, image_paths: list) -> dict:
    question = example.get("query") or ""
    response = example.get("response")
    answer_raw = (response or "").strip() if response is not None else ""
    if "choices" in example and example["choices"]:
        choices = list(example["choices"])
    elif "options" in example and example["options"]:
        choices = _parse_options(example["options"])
    else:
        choices = _choices_from_question(question)
    answer = _parse_answer(answer_raw, choices) if choices else answer_raw
    sub_task = example.get("sub_task") or example.get("dimension") or example.get("category") or ""
    return {
        "idx": idx,
        "images": image_paths,
        "dataset": "ZoomBench",
        "type": "multi-choice",
        "sub_task": sub_task,
        "question": question,
        "choices": choices,
        "answer": answer,
        "prompt": question,
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
    idx, image_dir, input_dir = args
    image_dir = Path(image_dir)
    dataset = load_dataset(input_dir, quiet=True)
    example = dataset[idx]
    images = extract_images(example)
    if not images:
        return None
    image_paths = save_images(images, idx, image_dir)
    return convert_sample(example, idx, image_paths)


def process_and_save(dataset, jsonl_path: Path, image_dir: Path, input_dir: str, num_proc: int = 8):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(i, image_dir, input_dir) for i in range(len(dataset))]
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
    if len(records) < len(dataset):
        print(f"Skipped {len(dataset) - len(records)} samples (no image or error)")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {jsonl_path}")
    print(f"Saved images to {image_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ZoomBench (MCQ only) to JSONL + image folder (BLINK schema)"
    )
    parser.add_argument("input_dir", type=str, help="Path to ZoomBench dir or hub name (e.g. inclusionAI/ZoomBench)")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--image_dir", type=str, required=True, help="Output image folder")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes")
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