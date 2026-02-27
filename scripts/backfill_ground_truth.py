#!/usr/bin/env python3
"""Backfill ground_truth from eval results into existing trace.json files.

Usage:
    python scripts/backfill_ground_truth.py \
        --memory_dir memory/train_run/ \
        --results_jsonl eval_results/run1/results.jsonl
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Backfill ground_truth from results.jsonl into trace.json files"
    )
    parser.add_argument("--memory_dir", type=str, required=True,
                        help="Memory directory containing tasks/*/trace.json")
    parser.add_argument("--results_jsonl", type=str, required=True,
                        help="Path to results.jsonl from evaluation run")
    args = parser.parse_args()

    tasks_dir = os.path.join(args.memory_dir, "tasks")
    if not os.path.exists(tasks_dir):
        print(f"Tasks directory not found: {tasks_dir}")
        sys.exit(1)

    # Load results: map dataset_id -> ground_truth
    gt_map = {}
    with open(args.results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            did = str(r.get("idx", ""))
            gt = r.get("ground_truth", "")
            if did and gt:
                gt_map[did] = gt

    print(f"Loaded {len(gt_map)} ground truth entries from {args.results_jsonl}")

    # Walk traces and backfill
    updated = 0
    skipped = 0
    for task_id in sorted(os.listdir(tasks_dir)):
        trace_path = os.path.join(tasks_dir, task_id, "trace.json")
        if not os.path.exists(trace_path):
            continue
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                trace = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if "ground_truth" in trace:
            skipped += 1
            continue

        did = str(trace.get("dataset_id", ""))
        if did not in gt_map:
            continue

        trace["ground_truth"] = gt_map[did]
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
        updated += 1

    print(f"Updated {updated} traces, skipped {skipped} (already had ground_truth)")


if __name__ == "__main__":
    main()
