"""Explore HuggingFace dataset structure and fields."""

import sys
import datasets
from pathlib import Path
from typing import Optional


def explore_dataset(
    dataset_path: str,
    split: str = "train",
    max_samples: int = 3,
):
    """Explore local HuggingFace dataset structure.
    
    Args:
        dataset_path: Local directory containing dataset files
        split: Dataset split to explore
        max_samples: Number of sample records to display
    """
    dataset_path = Path(dataset_path).expanduser()
    
    print(f"\n{'='*80}")
    print(f"📂 Path: {dataset_path}")
    print(f"📊 Split: {split}")
    print(f"{'='*80}\n")
    
    # Load dataset from local directory
    try:
        print("⏳ Loading...")
        ds = datasets.load_from_disk(str(dataset_path))
        
        # Handle split
        if isinstance(ds, datasets.DatasetDict):
            if split not in ds:
                print(f"❌ Split '{split}' not found. Available: {list(ds.keys())}\n")
                return
            ds = ds[split]
        
        print(f"✅ Loaded {len(ds)} records\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return
    
    # Show features
    print(f"📋 Features ({len(ds.features)}):")
    print("-" * 80)
    for name, dtype in ds.features.items():
        print(f"  • {name:25s}: {dtype}")
    print()
    
    # Show sample records
    print(f"📝 Sample Records (showing {min(max_samples, len(ds))}):")
    print("=" * 80)
    
    for i in range(min(max_samples, len(ds))):
        print(f"\n{'─'*80}")
        print(f"Record {i}")
        print(f"{'─'*80}")
        
        item = ds[i]
        for key, value in item.items():
            # Format different types
            if key in ["image", "images"]:
                if isinstance(value, list):
                    print(f"  {key:20s}: [{len(value)} images]")
                    for j, img in enumerate(value[:2]):
                        if hasattr(img, "size"):
                            print(f"    [{j}] {img.size} {img.mode}")
                elif hasattr(value, "size"):
                    print(f"  {key:20s}: {value.size} {value.mode}")
                else:
                    print(f"  {key:20s}: <Image>")
            elif key in ["video", "videos"]:
                if isinstance(value, list):
                    print(f"  {key:20s}: [{len(value)} videos]")
                else:
                    print(f"  {key:20s}: <Video>")
            elif isinstance(value, str) and len(value) > 120:
                print(f"  {key:20s}: {value[:120]}...")
            elif isinstance(value, list) and len(value) > 4:
                print(f"  {key:20s}: [list of {len(value)}] {value[:2]}...")
            else:
                print(f"  {key:20s}: {value}")
    
    print(f"\n{'='*80}")
    print("✅ Done")
    print(f"{'='*80}\n")


def list_splits(dataset_path: str):
    """List available splits for a dataset.
    
    Args:
        dataset_path: Local directory containing dataset files
    """
    dataset_path = Path(dataset_path).expanduser()
    
    print(f"\n📂 Path: {dataset_path}")
    print()
    
    try:
        ds = datasets.load_from_disk(str(dataset_path))
        
        if isinstance(ds, datasets.DatasetDict):
            print("📊 Available splits:")
            for name, split_ds in ds.items():
                print(f"  • {name:15s}: {len(split_ds):,} examples")
        else:
            print(f"📊 Single split with {len(ds):,} examples")
        print()
    except Exception as e:
        print(f"❌ Error: {e}\n")


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("=" * 80)
        print("Dataset Explorer")
        print("=" * 80)
        print("\nUsage:")
        print("  python explore_hf_dataset.py <local_path> [split] [max_samples]")
        print("  python explore_hf_dataset.py <local_path> --splits")
        print("\nExamples:")
        print("  python explore_hf_dataset.py ~/data/blink train 3")
        print("  python explore_hf_dataset.py ~/data/blink --splits")
        print("\n" + "=" * 80)
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    # Check if listing splits
    if len(sys.argv) > 2 and "--splits" in sys.argv[2:]:
        list_splits(dataset_path)
    else:
        split = sys.argv[2] if len(sys.argv) > 2 else "train"
        max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        explore_dataset(dataset_path, split, max_samples)


if __name__ == "__main__":
    main()
