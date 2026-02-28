"""Fix MMSI-Bench JSONL: move type -> sub_task, set type = 'multi-choice'.

Usage:
    python scripts/fix_mmsi_jsonl.py path/to/mmsi_bench.jsonl
"""

import json
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/fix_mmsi_jsonl.py <jsonl_path>")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    fixed = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    records = []
    for line in lines:
        record = json.loads(line)
        if record.get("dataset") == "MMSI-Bench" and record.get("type") != "multi-choice":
            record["sub_task"] = record["type"]
            record["type"] = "multi-choice"
            fixed += 1
        records.append(json.dumps(record, ensure_ascii=False))

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec + "\n")

    print(f"Fixed {fixed}/{len(records)} records in {jsonl_path}")


if __name__ == "__main__":
    main()
