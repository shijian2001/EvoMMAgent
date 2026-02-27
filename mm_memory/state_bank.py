"""State-level multi-view experience index over annotated trajectories."""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ALL_VIEWS = [
    "question", "task", "images", "observations",
    "question+task", "question+images", "question+observations",
    "task+images", "task+observations", "images+observations",
    "question+task+images", "question+task+observations",
    "question+images+observations", "task+images+observations",
    "all",
]

VIEW_DESCRIPTIONS: Dict[str, str] = {
    "question": "match by question text only",
    "task": "match by task type only",
    "images": "match by input images only (visual similarity)",
    "observations": "match by tool observations so far",
    "question+task": "match by question and task type",
    "question+images": "match by question and images (multimodal)",
    "question+observations": "match by question and observations",
    "task+images": "match by task type and images",
    "task+observations": "match by task type and observations",
    "images+observations": "match by images and observations",
    "question+task+images": "match by question, task, and images",
    "question+task+observations": "match by question, task, and observations",
    "question+images+observations": "match by question, images, and observations",
    "task+images+observations": "match by task, images, and observations",
    "all": "match by all elements (question, task, images, observations)",
}

_ELEMENT_NAMES = {"question", "task", "images", "observations"}
_OVER_FETCH_FACTOR = 5


def _view_parts(view: str) -> Set[str]:
    """Return the set of element names that a view includes."""
    if view == "all":
        return _ELEMENT_NAMES.copy()
    return set(view.split("+"))


def available_views(elements: Dict[str, Any]) -> List[str]:
    """Return views whose required elements are present in the state."""
    has: Set[str] = set()
    if elements.get("question"):
        has.add("question")
    if elements.get("task"):
        has.add("task")
    if elements.get("images"):
        has.add("images")
    if elements.get("observations"):
        has.add("observations")
    return [v for v in ALL_VIEWS if _view_parts(v).issubset(has)]


def compose_view(elements: Dict[str, Any], view: str) -> Dict[str, Any]:
    """Compose a retrieval query payload for one view.

    Returns:
        Dict with keys:
          - text: Text content for embedding
          - images: List[str] image paths used for multimodal embedding
    """
    parts_set = _view_parts(view)
    text_parts: List[str] = []
    images: List[str] = []

    if "question" in parts_set and elements.get("question"):
        text_parts.append(f"Question: {elements['question']}")
    if "task" in parts_set and elements.get("task"):
        text_parts.append(f"Task: {elements['task']}")
    if "images" in parts_set and elements.get("images"):
        images = [str(p) for p in elements["images"] if p]
        text_parts.append("Images are provided.")
    if "observations" in parts_set and elements.get("observations"):
        for obs in elements["observations"]:
            text_parts.append(f"Observation: {obs}")

    return {"text": "\n".join(text_parts), "images": images}


class StateBank:
    """State-level multi-view search index over hindsight-annotated trajectories."""

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
        views_dir = os.path.join(bank_dir, "views")

        if not os.path.exists(meta_path) or not os.path.exists(views_dir):
            raise FileNotFoundError(
                f"State bank not found at {bank_dir}. "
                f"Run scripts/build_state_bank.py first."
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            self.state_meta: List[Dict[str, Any]] = json.load(f)

        self.view_embeddings: Dict[str, np.ndarray] = {}
        self.view_masks: Dict[str, np.ndarray] = {}
        self._normed_by_view: Dict[str, np.ndarray] = {}

        for view in ALL_VIEWS:
            emb_path = os.path.join(views_dir, f"{view}.npy")
            if not os.path.exists(emb_path):
                continue
            emb = np.load(emb_path)
            if emb.shape[0] != len(self.state_meta):
                raise ValueError(
                    f"View {view} row count mismatch: "
                    f"{emb.shape[0]} vs meta {len(self.state_meta)}"
                )
            self.view_embeddings[view] = emb

            mask_path = os.path.join(views_dir, f"{view}_mask.npy")
            if os.path.exists(mask_path):
                mask = np.load(mask_path).astype(bool)
            else:
                mask = np.ones((emb.shape[0],), dtype=bool)
            self.view_masks[view] = mask

            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
            self._normed_by_view[view] = emb / norms

        logger.info(
            f"Loaded state bank: {len(self.state_meta)} states, "
            f"{len(self.view_embeddings)} views"
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_view(
        self,
        view: str,
        query_emb: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0,
        min_q: int = 0,
        exclude_ids: Optional[Set[Tuple[str, int]]] = None,
    ) -> List[Dict[str, Any]]:
        """Search a specific view by cosine similarity with Q-value filtering.

        Args:
            view: View identifier, one of ``ALL_VIEWS``
            query_emb: Query embedding, shape [1, D] or [D]
            top_k: Number of results to return
            min_score: Minimum cosine similarity threshold
            min_q: Minimum Q-value threshold; states below are discarded
            exclude_ids: Optional set of ``(task_id, state)`` to skip

        Returns:
            List of dicts with ``experience``, ``q_value``, ``retrieval_score``,
            ``state_text``, ``task_id``, ``state``, sorted by score desc.
        """
        if view not in self._normed_by_view:
            return []
        if len(self.state_meta) == 0:
            return []
        if exclude_ids is None:
            exclude_ids = set()

        # Normalize query
        query = query_emb.reshape(1, -1).astype(np.float32)
        query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity
        scores = (query @ self._normed_by_view[view].T).flatten()

        # Top-K indices (over-fetch to allow multiple filters)
        k = min(top_k * _OVER_FETCH_FACTOR, len(self.state_meta))
        top_indices = np.argsort(-scores, kind='stable')[:k]

        results = []
        seen_tasks: Set[str] = set()
        for idx in top_indices:
            if not self.view_masks[view][idx]:
                continue
            score = float(scores[idx])
            if score < min_score:
                break  # sorted desc
            meta = self.state_meta[idx]
            task_id = str(meta.get("task_id", ""))
            key = (task_id, int(meta.get("state", -1)))
            if key in exclude_ids:
                continue
            if meta.get("q_value", 0) < min_q:
                continue
            if task_id and task_id in seen_tasks:
                continue
            seen_tasks.add(task_id)
            result = dict(meta)
            result["retrieval_score"] = score
            result["view"] = view
            results.append(result)
            if len(results) >= top_k:
                break

        return results

    def search(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0,
        min_q: int = 0,
    ) -> List[Dict[str, Any]]:
        """Backward-compatible single-view search using ``all`` if available."""
        default_view = "all" if "all" in self.view_embeddings else (next(iter(self.view_embeddings), ""))
        if not default_view:
            return []
        return self.search_view(
            default_view,
            query_emb=query_emb,
            top_k=top_k,
            min_score=min_score,
            min_q=min_q,
        )

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    @staticmethod
    def state_to_elements(
        trace_data: Dict[str, Any],
        trajectory: List[Dict[str, Any]],
        step_index: int,
    ) -> Dict[str, Any]:
        """Serialize state ``s_t`` into structured elements for multi-view retrieval."""
        query = str(trace_data.get("input", {}).get("question", ""))
        sub_task = str(trace_data.get("sub_task", ""))

        input_images = trace_data.get("input", {}).get("images", [])
        image_paths: List[str] = []
        for img in input_images:
            if isinstance(img, dict):
                p = img.get("path")
            else:
                p = img
            if p:
                image_paths.append(str(p))

        observations: List[str] = []
        for i in range(step_index):
            a = trajectory[i]["action"]
            tool = a.get("tool", "")
            if tool == "answer":
                continue
            obs = a.get("observation") or ""
            if isinstance(obs, dict):
                obs = obs.get("description", str(obs))
            params_str = str(a.get("parameters", {}))
            observations.append(f"{tool}({params_str}) -> {obs}")

        return {
            "question": query,
            "task": sub_task,
            "images": image_paths,
            "observations": observations,
        }

    @staticmethod
    def state_to_text(
        trace_data: Dict[str, Any],
        trajectory: List[Dict[str, Any]],
        step_index: int,
    ) -> str:
        """Text serializer using the full ``all`` view."""
        elements = StateBank.state_to_elements(trace_data, trajectory, step_index)
        return compose_view(elements, "all").get("text", "")
