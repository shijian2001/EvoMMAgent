"""Trace bank: pre-built multimodal index over trace-level experiences."""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class TraceBank:
    """Search index over precomputed trace-level experiences."""

    def __init__(self, bank_dir: str):
        embeddings_path = os.path.join(bank_dir, "embeddings.npy")
        experiences_path = os.path.join(bank_dir, "experiences.json")

        if not os.path.exists(embeddings_path) or not os.path.exists(experiences_path):
            raise FileNotFoundError(
                f"Trace bank not found at {bank_dir}. "
                f"Run scripts/build_trace_bank.py first."
            )

        self.embeddings: np.ndarray = np.load(embeddings_path)
        with open(experiences_path, "r", encoding="utf-8") as f:
            raw: List[Union[str, Dict[str, Any]]] = json.load(f)

        # Normalize: old format (List[str]) -> new format (List[Dict])
        self.experiences: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, str):
                self.experiences.append({"experience": item, "source": "correct", "task_id": ""})
            else:
                self.experiences.append(item)

        if self.embeddings.shape[0] != len(self.experiences):
            raise ValueError(
                f"embeddings ({self.embeddings.shape[0]}) and experiences "
                f"({len(self.experiences)}) count mismatch"
            )

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        self._normed_embeddings = self.embeddings / norms

        logger.info(
            f"Loaded trace bank: {len(self.experiences)} entries, "
            f"embedding dim={self.embeddings.shape[1]}"
        )

    def search(self, query_emb: np.ndarray, min_score: float = 0.0) -> Optional[Dict[str, Any]]:
        """Search bank by cosine similarity and return top-1 experience dict."""
        if len(self.experiences) == 0:
            return None

        query = query_emb.reshape(1, -1).astype(np.float32)
        query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
        scores = (query @ self._normed_embeddings.T).flatten()

        top_idx = int(np.argmax(scores))
        top_score = float(scores[top_idx])
        if top_score < min_score:
            return None

        logger.info(f"Trace retrieval hit: score={top_score:.4f}")
        return self.experiences[top_idx]

    @staticmethod
    def build_index_text(trace: Dict) -> str:
        """Build text used for multimodal embedding."""
        question = trace.get("input", {}).get("question", "")
        sub_task = trace.get("sub_task", "")

        parts: List[str] = []
        if question:
            parts.append(f"Question: {question}")
        if sub_task:
            parts.append(f"Task: {sub_task}")
        return "\n".join(parts)
