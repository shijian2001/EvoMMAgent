"""Preprocess BLINK dataset to unified parquet format."""

import argparse
from pathlib import Path

import datasets
from datasets import Features, Sequence, Value, Image


def parse_answer(answer_raw: str, choices: list) -> str:
    """Map answer like '(A)' or 'A' to actual choice text."""
    if answer_raw == "hidden":
        return ""
    letter = answer_raw.strip("()").upper()
    idx = ord(letter) - ord("A") if letter.isalpha() and len(letter) == 1 else -1
    return choices[idx] if 0 <= idx < len(choices) else answer_raw


def process_fn(example: dict, idx: int) -> dict:
    """Convert a BLINK sample to unified format."""
    images = [example[f"image_{i}"] for i in range(1, 5) if example[f"image_{i}"] is not None]

    return {
        "idx": idx,
        "images": images,
        "dataset": "BLINK",
        "type": "multi-choice",
        "sub_task": example["sub_task"],
        "question": example["question"],
        "choices": example["choices"],
        "answer": parse_answer(example["answer"], example["choices"]),
        "prompt": example["prompt"],
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess BLINK dataset to unified format")
    parser.add_argument("input_dir", type=str, help="Path to BLINK dataset directory")
    parser.add_argument("output_dir", type=str, help="Output directory for parquet file")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes for parallel processing")
    args = parser.parse_args()

    # Load all validation splits and concatenate
    configs = datasets.get_dataset_config_names(args.input_dir)
    print(f"📂 Found {len(configs)} configs")

    combined = datasets.concatenate_datasets([
        datasets.load_dataset(args.input_dir, name=cfg, split="val")
        for cfg in configs
    ])
    print(f"📊 Total: {len(combined)} samples")

    # Define target schema
    features = Features({
        "idx": Value("int64"),
        "images": Sequence(Image()),
        "dataset": Value("string"),
        "type": Value("string"),
        "sub_task": Value("string"),
        "question": Value("string"),
        "choices": Sequence(Value("string")),
        "answer": Value("string"),
        "prompt": Value("string"),
    })

    # Process with map and cast to target schema
    processed = combined.map(
        process_fn,
        with_indices=True,
        remove_columns=combined.column_names,
        num_proc=args.num_proc,
    ).cast(features)

    # Save to parquet
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eval_data.parquet"
    processed.to_parquet(str(output_path))
    print(f"✅ Saved {len(processed)} samples to {output_path}")


if __name__ == "__main__":
    main()
