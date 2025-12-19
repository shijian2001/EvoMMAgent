#!/usr/bin/env python3
"""Explore local HF dataset structure (downloaded via `hf cli`)."""

import sys
from datasets import load_from_disk, load_dataset
from pathlib import Path


def truncate(val, max_len=150):
    """Truncate value repr for display."""
    s = repr(val)
    return s[:max_len] + "..." if len(s) > max_len else s


def explore(path: str):
    """Explore dataset and print first sample from ALL config/split combinations."""
    p = Path(path)
    print(f"\n🔍 Exploring: {p}\n{'=' * 60}")

    # Detect configs (subdirs containing data files)
    config_dirs = [d for d in p.iterdir() if d.is_dir() and not d.name.startswith('.')]
    has_configs = config_dirs and any(
        (d / "dataset_info.json").exists() or list(d.glob("*.arrow")) or list(d.glob("*.parquet"))
        for d in config_dirs
    )
    configs = [(d.name, d) for d in config_dirs] if has_configs else [("default", p)]

    print(f"📁 Configs: {[c[0] for c in configs]}\n")

    # Iterate ALL config/split combinations
    for config_name, config_path in configs:
        try:
            try:
                ds = load_from_disk(str(config_path))
            except:
                ds = load_dataset(str(config_path))

            splits = list(ds.keys()) if hasattr(ds, "keys") else ["data"]
            if not hasattr(ds, "keys"):
                ds = {"data": ds}

            for split in splits:
                data = ds[split]
                print(f"{'─' * 60}")
                print(f"📂 {config_name} / {split}  (rows={len(data):,})")
                print(f"   Features: {list(data.features.keys())}")
                print(f"   Sample[0]:")
                for k, v in data[0].items():
                    print(f"      • {k}: {truncate(v)}")

        except Exception as e:
            print(f"{'─' * 60}")
            print(f"📂 {config_name} ❌ Error: {e}")

    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <local_dataset_path>")
        sys.exit(1)
    explore(sys.argv[1])