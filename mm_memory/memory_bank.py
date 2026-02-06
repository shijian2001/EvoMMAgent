"""Memory bank: a thin search index over existing memory traces.

The bank lives at ``{memory_dir}/bank/`` and contains only:
- ``embeddings.npy``: [N, D] float32 vectors
- ``task_ids.json``: ordered list of task IDs matching embedding rows

All trace data is loaded on-demand from ``{memory_dir}/tasks/{task_id}/trace.json``.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MemoryBank:
    """Search index over existing memory traces."""

    def __init__(self, memory_dir: str):
        """Load a pre-built bank from ``{memory_dir}/bank/``.

        Args:
            memory_dir: Root memory directory containing ``tasks/`` and ``bank/``

        Raises:
            FileNotFoundError: If bank files do not exist
        """
        self.memory_dir = memory_dir
        bank_dir = os.path.join(memory_dir, "bank")

        task_ids_path = os.path.join(bank_dir, "task_ids.json")
        embeddings_path = os.path.join(bank_dir, "embeddings.npy")

        if not os.path.exists(task_ids_path) or not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"Memory bank not found at {bank_dir}. "
                f"Run scripts/build_memory_bank.py first."
            )

        with open(task_ids_path, "r", encoding="utf-8") as f:
            self.task_ids: List[str] = json.load(f)

        self.embeddings: np.ndarray = np.load(embeddings_path)

        if len(self.task_ids) != self.embeddings.shape[0]:
            raise ValueError(
                f"task_ids ({len(self.task_ids)}) and embeddings "
                f"({self.embeddings.shape[0]}) count mismatch"
            )

        # Pre-normalize embeddings for fast cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        self._normed_embeddings = self.embeddings / norms

        logger.info(
            f"Loaded memory bank: {len(self.task_ids)} entries, "
            f"embedding dim={self.embeddings.shape[1]}"
        )

    def search(self, query_emb: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search bank by cosine similarity.

        Args:
            query_emb: Query embedding, shape [1, D] or [D]
            top_k: Number of results to return

        Returns:
            List of trace dicts (loaded from trace.json) with added
            ``retrieval_score`` and ``task_id`` fields, sorted by score desc.
        """
        if len(self.task_ids) == 0:
            return []

        # Normalize query
        query = query_emb.reshape(1, -1).astype(np.float32)
        query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity
        scores = (query @ self._normed_embeddings.T).flatten()

        # Top-K indices
        k = min(top_k, len(self.task_ids))
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            task_id = self.task_ids[idx]
            trace = self._load_trace(task_id)
            if trace is not None:
                trace["retrieval_score"] = float(scores[idx])
                trace["task_id"] = task_id
                results.append(trace)

        return results

    def _load_trace(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load trace.json for a given task on demand.

        Args:
            task_id: Task ID (e.g. "000042")

        Returns:
            Parsed trace dict, or None if file is missing/corrupt
        """
        trace_path = os.path.join(self.memory_dir, "tasks", task_id, "trace.json")
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load trace for task {task_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Offline construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_index_text(trace: Dict[str, Any]) -> str:
        """Build the text representation used for embedding a trace.

        Concatenates question, sub_task, tools used, and answer.
        The resulting text is used to compute the embedding and is NOT stored.

        Args:
            trace: Parsed trace.json dict

        Returns:
            A single text string for embedding
        """
        parts: List[str] = []

        # Question
        question = trace.get("input", {}).get("question", "")
        if question:
            parts.append(question)

        # Sub-task type
        sub_task = trace.get("sub_task", "")
        if sub_task:
            parts.append(f"Task: {sub_task}")

        # Tools used
        tools = [
            step["tool"]
            for step in trace.get("trace", [])
            if step.get("type") == "action" and "tool" in step
        ]
        if tools:
            parts.append(f"Tools: {', '.join(tools)}")

        # Answer
        answer = trace.get("answer", "")
        if answer:
            parts.append(f"Answer: {answer}")

        return ". ".join(parts)

    @classmethod
    async def build(
        cls,
        memory_dir: str,
        embedder,
        filter_correct: bool = True,
        batch_size: int = 32,
    ) -> "MemoryBank":
        """Build a memory bank from existing traces (offline).

        Scans ``{memory_dir}/tasks/*/trace.json``, filters, embeds,
        and saves the bank to ``{memory_dir}/bank/``.

        Args:
            memory_dir: Root memory directory
            embedder: An ``Embedder`` instance
            filter_correct: If True, only include traces with ``is_correct=True``
            batch_size: Batch size for embedding API calls

        Returns:
            The built MemoryBank instance
        """
        tasks_dir = os.path.join(memory_dir, "tasks")
        if not os.path.exists(tasks_dir):
            raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

        # Scan and filter traces
        task_ids: List[str] = []
        texts: List[str] = []

        task_dirs = sorted(
            d
            for d in os.listdir(tasks_dir)
            if os.path.isdir(os.path.join(tasks_dir, d))
        )

        for task_id in task_dirs:
            trace_path = os.path.join(tasks_dir, task_id, "trace.json")
            if not os.path.exists(trace_path):
                continue

            try:
                with open(trace_path, "r", encoding="utf-8") as f:
                    trace = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Skipping {task_id}: {e}")
                continue

            if filter_correct and not trace.get("is_correct", False):
                continue

            index_text = cls.build_index_text(trace)
            if not index_text.strip():
                continue

            task_ids.append(task_id)
            texts.append(index_text)

        if not task_ids:
            raise ValueError(
                f"No valid traces found in {tasks_dir} "
                f"(filter_correct={filter_correct})"
            )

        logger.info(f"Building memory bank from {len(task_ids)} traces ...")

        # Embed
        embeddings = await embedder.encode_batch(texts, batch_size=batch_size)

        # Save
        bank_dir = os.path.join(memory_dir, "bank")
        os.makedirs(bank_dir, exist_ok=True)

        np.save(os.path.join(bank_dir, "embeddings.npy"), embeddings)
        with open(os.path.join(bank_dir, "task_ids.json"), "w", encoding="utf-8") as f:
            json.dump(task_ids, f, ensure_ascii=False)

        logger.info(
            f"Memory bank saved to {bank_dir}: "
            f"{len(task_ids)} entries, dim={embeddings.shape[1]}"
        )

        return cls(memory_dir)
