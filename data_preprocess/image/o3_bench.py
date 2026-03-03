"""Preprocess O3-Bench dataset to JSONL + image folder format."""
'''
python o3_bench.py /path/to/O3-Bench --jsonl_path out/o3_bench.jsonl --image_dir out/o3_bench_images --num_proc 8

'''

import argparse
import io
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import datasets
from PIL import Image
from tqdm import tqdm


def load_dataset(input_dir: str, quiet: bool = False):
    input_path = Path(input_dir)
    if input_path.is_dir() and (input_path / "dataset_infos.json").exists():
        ds = datasets.load_dataset(str(input_path), split="test")
    else:
        ds = datasets.load_dataset(input_dir, split="test")
    if not quiet:
        print(f"Loaded {len(ds)} samples")
    return ds


def extract_images(example: dict) -> list:
    img = example.get("image")
    if img is None:
        return []
    if hasattr(img, "save"):
        return [img.convert("RGB") if img.mode != "RGB" else img]
    if isinstance(img, bytes):
        try:
            return [Image.open(io.BytesIO(img)).convert("RGB")]
        except Exception:
            return []
    if isinstance(img, dict) and "bytes" in img:
        try:
            return [Image.open(io.BytesIO(img["bytes"])).convert("RGB")]
        except Exception:
            return []
    return []


def _parse_options(options_str: str) -> list:
    if not options_str or not isinstance(options_str, str):
        return []
    options_str = options_str.strip()
    pattern = re.compile(
        r"\(?([A-F])\)?\s*[.:]\s*(.*?)(?=\s*\(?[A-F]\)?\s*[.:]\s*|\s*$)",
        re.DOTALL | re.IGNORECASE,
    )
    matches = pattern.findall(options_str)
    if matches:
        return [m[1].strip() for m in matches]
    lines = [ln.strip() for ln in options_str.split("\n") if ln.strip()]
    result = []
    for line in lines:
        m = re.match(r"^\(?([A-F])\)?\s*[.:]\s*(.*)$", line, re.IGNORECASE)
        if m:
            result.append(m.group(2).strip())
    return result if result else []


def _parse_answer(answer_raw: str, choices: list) -> str:
    if answer_raw is None or (isinstance(answer_raw, str) and answer_raw.strip() == ""):
        return ""
    letter = str(answer_raw).strip("()").upper()
    if not letter or len(letter) > 1:
        return str(answer_raw)
    idx = ord(letter) - ord("A") if letter.isalpha() else -1
    return choices[idx] if 0 <= idx < len(choices) else str(answer_raw)


def convert_sample(example: dict, idx: int, image_paths: list[str]) -> dict:
    question = example.get("question") or ""
    options_str = example.get("options") or ""
    choices = _parse_options(options_str)
    answer_raw = example.get("answer") or ""
    return {
        "idx": idx,
        "images": image_paths,
        "dataset": "O3-Bench",
        "type": "multi-choice",
        "sub_task": example.get("subset") or "",
        "question": question,
        "choices": choices,
        "answer": _parse_answer(answer_raw, choices),
        "prompt": "",
    }


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
    idx, image_dir, input_dir = args
    image_dir = Path(image_dir)
    ds = load_dataset(input_dir, quiet=True)
    example = ds[idx]
    images = extract_images(example)
    if not images:
        return None
    image_paths = save_images(images, idx, image_dir)
    return convert_sample(example, idx, image_paths)


def process_and_save(dataset, jsonl_path: Path, image_dir: Path, input_dir: str, num_proc: int = 8):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    tasks = [(i, image_dir, input_dir) for i in range(n)]
    print(f"Processing with {num_proc} processes...")
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        records = list(
            tqdm(
                executor.map(process_one_sample, tasks),
                total=n,
                desc="Processing samples",
            )
        )
    records = [r for r in records if r is not None]
    if len(records) < n:
        print(f"Skipped {n - len(records)} samples (no image or error)")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {jsonl_path}")
    print(f"Saved images to {image_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess O3-Bench to JSONL + image folder (BLINK schema)"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to O3-Bench dir or HuggingFace hub name (e.g. InSight-o3/O3-Bench)",
    )
    parser.add_argument("--jsonl_path", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--image_dir", type=str, required=True, help="Output image folder")
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