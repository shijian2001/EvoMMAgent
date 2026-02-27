"""State-level experience index over annotated trajectories.

The bank lives at ``{memory_dir}/state_bank/`` and contains:
- ``embeddings.npy``:   [M, D] float32 vectors (one per state)
- ``state_meta.json``:  ordered list of dicts matching embedding rows,
  each with ``task_id``, ``state``, ``experience``, ``q_value``, ``state_text``
"""

import json
import logging
import os
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class StateBank:
    """State-level search index over hindsight-annotated trajectories."""

    def __init__(self, memory_dir: str, bank_dir_name: str = "state_bank"):
        """Load a pre-built state bank from ``{memory_dir}/{bank_dir_name}/``.

        Args:
            memory_dir: Root memory directory containing ``tasks/`` and the bank subfolder.
            bank_dir_name: Name of the bank subfolder (default ``state_bank``).

        Raises:
            FileNotFoundError: If state bank files do not exist
        """
        self.memory_dir = memory_dir
        bank_dir = os.path.join(memory_dir, bank_dir_name)

        meta_path = os.path.join(bank_dir, "state_meta.json")
        embeddings_path = os.path.join(bank_dir, "embeddings.npy")

        if not os.path.exists(meta_path) or not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"State bank not found at {bank_dir}. "
                f"Run scripts/build_state_bank.py first."
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            self.state_meta: List[Dict[str, Any]] = json.load(f)

        self.embeddings: np.ndarray = np.load(embeddings_path)

        if len(self.state_meta) != self.embeddings.shape[0]:
            raise ValueError(
                f"state_meta ({len(self.state_meta)}) and embeddings "
                f"({self.embeddings.shape[0]}) count mismatch"
            )

        # Pre-normalize for fast cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        self._normed_embeddings = self.embeddings / norms

        logger.info(
            f"Loaded state bank: {len(self.state_meta)} states, "
            f"embedding dim={self.embeddings.shape[1]}"
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0,
        min_q: int = 0,
    ) -> List[Dict[str, Any]]:
        """Search bank by cosine similarity with Q-value filtering.

        Args:
            query_emb: Query embedding, shape [1, D] or [D]
            top_k: Number of results to return
            min_score: Minimum cosine similarity threshold
            min_q: Minimum Q-value threshold; states below are discarded

        Returns:
            List of dicts with ``experience``, ``q_value``, ``retrieval_score``,
            ``state_text``, ``task_id``, ``state``, sorted by score desc.
        """
        if len(self.state_meta) == 0:
            return []

        # Normalize query
        query = query_emb.reshape(1, -1).astype(np.float32)
        query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity
        scores = (query @ self._normed_embeddings.T).flatten()

        # Top-K indices
        k = min(top_k * 3, len(self.state_meta))  # over-fetch to allow Q filtering
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                break  # sorted desc
            meta = self.state_meta[idx]
            if meta.get("q_value", 0) < min_q:
                continue
            result = dict(meta)
            result["retrieval_score"] = score
            results.append(result)
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    @staticmethod
    def state_to_text(
        trace_data: Dict[str, Any],
        trajectory: List[Dict[str, Any]],
        step_index: int,
        image_caption: str = "",
    ) -> str:
        """Serialize state s_t into text for embedding.

        s_0 = image caption + question (+ sub_task).
        s_t = s_0 + summary of a_0 ... a_{t-1}.

        Args:
            trace_data: Parsed trace.json dict (needs ``input.question``, optional ``sub_task``)
            trajectory: List of trajectory entries (each has ``action`` dict)
            step_index: Current step index (0 = initial state)
            image_caption: Optional 1-2 sentence image description for the task

        Returns:
            Text representation of state s_t
        """
        query = trace_data.get("input", {}).get("question", "")
        sub_task = trace_data.get("sub_task", "")

        parts = []
        if image_caption:
            parts.append(f"Image description: {image_caption}")
        if query:
            parts.append(f"Question: {query}")
        if sub_task:
            parts.append(f"Task: {sub_task}")

        for i in range(step_index):
            a = trajectory[i]["action"]
            if a["tool"] == "answer":  # terminal action: no observation to include in state text
                continue
            obs = a.get("observation") or ""
            if isinstance(obs, dict):
                obs = obs.get("description", str(obs))
            params_str = str(a.get("parameters", {}))
            parts.append(f"{a['tool']}({params_str}) → {obs}")

        return "\n".join(parts)

    # Note: Offline construction is handled by scripts/build_state_bank.py
    # which does annotate → extract → embed → save in a single in-memory pipeline.
