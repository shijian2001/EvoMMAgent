#!/usr/bin/env python3
"""Merge multiple state banks into one unified bank.

Equivalent to merging all traces first and building a single bank,
since annotation and embedding are per-state independent operations.

Usage:
    python scripts/merge_state_banks.py \
        --banks vstar:memory/vstar_train/state_bank \
                mmvet:memory/mmvet_train/state_bank \
                mathvista:memory/mathvista_train/state_bank \
        --output memory/merged/state_bank

    Each --banks entry is  prefix:path  where prefix is prepended to
    task_id to avoid collisions across datasets.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm_memory.state_bank import ALL_VIEWS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_bank_arg(arg: str) -> Tuple[str, str]:
    """Parse 'prefix:path' into (prefix, path)."""
    if ":" not in arg:
        raise ValueError(f"Expected 'prefix:path', got '{arg}'")
    prefix, path = arg.split(":", 1)
    return prefix.strip(), path.strip()


def merge_state_banks(bank_specs: List[Tuple[str, str]], output_dir: str) -> None:
    merged_meta: List[Dict] = []
    per_bank_sizes: List[int] = []
    embed_dim: int = 0

    # ── Pass 1: load and merge state_meta ──
    for prefix, bank_dir in bank_specs:
        meta_path = os.path.join(bank_dir, "state_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"state_meta.json not found in {bank_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            metas = json.load(f)

        for m in metas:
            m["task_id"] = f"{prefix}_{m['task_id']}"

        per_bank_sizes.append(len(metas))
        merged_meta.extend(metas)
        logger.info(f"  [{prefix}] {len(metas)} states from {bank_dir}")

    total = len(merged_meta)
    logger.info(f"Total: {total} states from {len(bank_specs)} banks")

    # ── Pass 2: merge per-view embeddings and masks ──
    os.makedirs(os.path.join(output_dir, "views"), exist_ok=True)
    views_merged = 0

    for view in ALL_VIEWS:
        embs: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        all_exist = True

        for (prefix, bank_dir), n_states in zip(bank_specs, per_bank_sizes):
            emb_path = os.path.join(bank_dir, "views", f"{view}.npy")
            mask_path = os.path.join(bank_dir, "views", f"{view}_mask.npy")

            if os.path.exists(emb_path):
                emb = np.load(emb_path)
                assert emb.shape[0] == n_states, (
                    f"[{prefix}] {view}.npy has {emb.shape[0]} rows, "
                    f"expected {n_states} (state_meta length)"
                )
                if embed_dim == 0:
                    embed_dim = emb.shape[1]
                else:
                    assert emb.shape[1] == embed_dim, (
                        f"[{prefix}] {view}.npy dim={emb.shape[1]}, expected {embed_dim}"
                    )
                embs.append(emb)

                if os.path.exists(mask_path):
                    mask = np.load(mask_path).astype(bool)
                    assert mask.shape[0] == n_states
                else:
                    mask = np.ones(n_states, dtype=bool)
                masks.append(mask)
            else:
                all_exist = False
                embs.append(None)
                masks.append(None)

        has_any = any(e is not None for e in embs)
        if not has_any:
            continue

        merged_emb = np.zeros((total, embed_dim), dtype=np.float32)
        merged_mask = np.zeros(total, dtype=bool)
        offset = 0
        for emb, mask, n in zip(embs, masks, per_bank_sizes):
            if emb is not None:
                merged_emb[offset : offset + n] = emb
                merged_mask[offset : offset + n] = mask
            offset += n

        np.save(os.path.join(output_dir, "views", f"{view}.npy"), merged_emb)
        np.save(os.path.join(output_dir, "views", f"{view}_mask.npy"), merged_mask)
        views_merged += 1

        valid = int(merged_mask.sum())
        logger.info(f"  View {view}: {valid}/{total} valid states")

    # ── Save merged meta ──
    with open(os.path.join(output_dir, "state_meta.json"), "w", encoding="utf-8") as f:
        json.dump(merged_meta, f, ensure_ascii=False, indent=2)

    logger.info(
        f"\nMerged bank saved to {output_dir}: "
        f"{total} states, {views_merged} views, dim={embed_dim}"
    )

    # ── Verification ──
    logger.info("Verifying merged bank...")
    from mm_memory.state_bank import StateBank
    bank = StateBank.__new__(StateBank)
    bank.memory_dir = os.path.dirname(output_dir)
    with open(os.path.join(output_dir, "state_meta.json")) as f:
        bank.state_meta = json.load(f)
    for view in ALL_VIEWS:
        emb_path = os.path.join(output_dir, "views", f"{view}.npy")
        if os.path.exists(emb_path):
            emb = np.load(emb_path)
            assert emb.shape[0] == len(bank.state_meta), (
                f"Verification failed: {view}.npy rows={emb.shape[0]} != meta={len(bank.state_meta)}"
            )
    logger.info("Verification passed ✓")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple state banks.")
    parser.add_argument(
        "--banks", nargs="+", required=True,
        help="List of prefix:bank_dir pairs (e.g. vstar:memory/vstar/state_bank)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for the merged bank",
    )
    args = parser.parse_args()

    bank_specs = [parse_bank_arg(b) for b in args.banks]
    merge_state_banks(bank_specs, args.output)


if __name__ == "__main__":
    main()
