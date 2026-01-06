"""Dataset slicer for creating balanced subsets from JSONL datasets."""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: List[Dict], file_path: str):
    """Save data to JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def sample_random(data: List[Dict], n: int) -> List[Dict]:
    """Random sampling."""
    return random.sample(data, min(n, len(data)))


def sample_balanced(data: List[Dict], n: int, key: str = "sub_task") -> List[Dict]:
    """Balanced sampling by sub_task."""
    # Group by sub_task
    groups = defaultdict(list)
    for item in data:
        groups[item.get(key, "unknown")].append(item)
    
    # Calculate samples per group
    n_groups = len(groups)
    per_group = n // n_groups
    remainder = n % n_groups
    
    # Sample from each group
    sampled = []
    for i, (task, items) in enumerate(sorted(groups.items())):
        n_samples = per_group + (1 if i < remainder else 0)
        sampled.extend(random.sample(items, min(n_samples, len(items))))
    
    # Shuffle final result
    random.shuffle(sampled)
    return sampled[:n]


def main():
    parser = argparse.ArgumentParser(description="Slice JSONL dataset into balanced subsets")
    parser.add_argument("input", help="Input JSONL file path")
    parser.add_argument("output", help="Output JSONL file path")
    parser.add_argument("-n", "--num_samples", type=int, required=True,
                       help="Number of samples to extract")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load data
    print(f"📂 Loading {args.input}...")
    data = load_jsonl(args.input)
    print(f"   Total samples: {len(data)}")
    
    # Check if sub_task exists
    has_subtask = any(item.get("sub_task") for item in data)
    
    # Sample
    if has_subtask:
        print(f"🔀 Balanced sampling by sub_task...")
        sampled = sample_balanced(data, args.num_samples)
        
        # Print distribution
        subtask_counts = defaultdict(int)
        for item in sampled:
            subtask_counts[item.get("sub_task", "unknown")] += 1
        print("   Distribution:")
        for task, count in sorted(subtask_counts.items()):
            print(f"     - {task}: {count}")
    else:
        print(f"🎲 Random sampling...")
        sampled = sample_random(data, args.num_samples)
    
    # Save
    print(f"💾 Saving to {args.output}...")
    save_jsonl(sampled, args.output)
    print(f"✅ Done! Extracted {len(sampled)} samples")


if __name__ == "__main__":
    main()

