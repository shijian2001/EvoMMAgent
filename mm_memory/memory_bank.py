"""Memory bank: a thin search index over existing memory traces.

The bank lives at ``{memory_dir}/trace_bank/`` and contains:
- ``embeddings.npy``: [N, D] float32 vectors
- ``task_ids.json``: ordered list of task IDs matching embedding rows
- ``captions.json``: ordered list of image captions (empty string if no images)

All trace data is loaded on-demand from ``{memory_dir}/tasks/{task_id}/trace.json``.
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CAPTION_PROMPT = (
    "Describe what these images show as a whole in 1-2 concise sentences "
    "(under 30 words). Focus on the visual content and task scenario."
)


class MemoryBank:
    """Search index over existing memory traces."""

    def __init__(self, memory_dir: str, bank_dir_name: str = "trace_bank"):
        """Load a pre-built bank from ``{memory_dir}/{bank_dir_name}/``.

        Args:
            memory_dir: Root memory directory containing ``tasks/`` and the bank subfolder.
            bank_dir_name: Name of the bank subfolder (default ``trace_bank``).

        Raises:
            FileNotFoundError: If bank files do not exist
        """
        self.memory_dir = memory_dir
        bank_dir = os.path.join(memory_dir, bank_dir_name)

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

        # Load captions (optional — backward compatible with older banks)
        captions_path = os.path.join(bank_dir, "captions.json")
        if os.path.exists(captions_path):
            with open(captions_path, "r", encoding="utf-8") as f:
                self.captions: List[str] = json.load(f)
        else:
            self.captions = [""] * len(self.task_ids)

        # Pre-normalize embeddings for fast cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        self._normed_embeddings = self.embeddings / norms

        logger.info(
            f"Loaded memory bank: {len(self.task_ids)} entries, "
            f"embedding dim={self.embeddings.shape[1]}"
        )

    def search(
        self, query_emb: np.ndarray, top_k: int = 10, min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search bank by cosine similarity.

        Args:
            query_emb: Query embedding, shape [1, D] or [D]
            top_k: Number of results to return
            min_score: Minimum cosine similarity threshold; results below
                this score are discarded.

        Returns:
            List of trace dicts (loaded from trace.json) with added
            ``retrieval_score``, ``task_id``, and ``_caption`` fields,
            sorted by score desc.
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
            score = float(scores[idx])
            if score < min_score:
                break  # sorted desc, remaining are even lower
            task_id = self.task_ids[idx]
            trace = self._load_trace(task_id)
            if trace is not None:
                trace["retrieval_score"] = score
                trace["task_id"] = task_id
                trace["_caption"] = self.captions[idx] if idx < len(self.captions) else ""
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
    # Index text construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_index_text(trace: Dict[str, Any], caption: str = "") -> str:
        """Build the text representation used for embedding a trace.

        Concatenates caption, question, sub_task, and tools (deduplicated,
        in first-occurrence order).  Answer is intentionally excluded —
        retrieval targets strategy similarity, not answer similarity.

        Args:
            trace: Parsed trace.json dict
            caption: Optional image caption for this trace

        Returns:
            A single text string for embedding
        """
        parts: List[str] = []

        # Image caption
        if caption:
            parts.append(f"Image description: {caption}")

        # Question
        question = trace.get("input", {}).get("question", "")
        if question:
            parts.append(f"Question: {question}")

        # Sub-task type
        sub_task = trace.get("sub_task", "")
        if sub_task:
            parts.append(f"Task: {sub_task}")

        # Tools used — deduplicated, preserving first-occurrence order
        seen: set = set()
        tools_ordered: List[str] = []
        for step in trace.get("trace", []):
            if step.get("type") == "action" and "tool" in step:
                t = step["tool"]
                if t not in seen:
                    seen.add(t)
                    tools_ordered.append(t)
        if tools_ordered:
            parts.append(f"Tools (in order): {', '.join(tools_ordered)}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Caption generation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _image_to_data_uri(path: str) -> str:
        """Convert a local image file to a base64 data URI.

        Args:
            path: Local file path to an image

        Returns:
            ``data:image/<mime>;base64,...`` string
        """
        mime, _ = mimetypes.guess_type(path)
        if not mime or not mime.startswith("image/"):
            mime = "image/jpeg"

        with open(path, "rb") as f:
            img_bytes = f.read()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    @staticmethod
    async def _generate_caption(api_pool, image_paths: List[str]) -> str:
        """Generate a concise caption for input images via VLM.

        Args:
            api_pool: APIPool instance with a multimodal model
            image_paths: List of image file paths

        Returns:
            Caption string, or empty string on failure
        """
        content: list = [{"type": "text", "text": CAPTION_PROMPT}]
        for p in image_paths:
            data_uri = MemoryBank._image_to_data_uri(p)
            content.append({"type": "image_url", "image_url": {"url": data_uri}})

        try:
            result = await api_pool.execute(
                "qa",
                system="You are a concise image captioner.",
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
                max_tokens=100,
            )
            return result.get("answer", "").strip()
        except Exception as e:
            logger.warning(f"Caption generation failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # Offline construction
    # ------------------------------------------------------------------

    @classmethod
    async def build(
        cls,
        memory_dir: str,
        embedder,
        api_pool=None,
        filter_correct: bool = True,
        batch_size: int = 32,
        bank_dir_name: str = "trace_bank",
    ) -> "MemoryBank":
        """Build a memory bank from existing traces (offline).

        Scans ``{memory_dir}/tasks/*/trace.json``, filters, optionally
        generates image captions via *api_pool*, computes embeddings,
        and saves the bank to ``{memory_dir}/{bank_dir_name}/``.

        Args:
            memory_dir: Root memory directory
            embedder: An ``Embedder`` instance
            api_pool: Optional ``APIPool`` for caption generation.
                      When ``None``, captions are skipped (all empty).
            filter_correct: If True, only include traces with ``is_correct=True``
            batch_size: Batch size for embedding API calls
            bank_dir_name: Name of the output subfolder under *memory_dir*.

        Returns:
            The built MemoryBank instance
        """
        tasks_dir = os.path.join(memory_dir, "tasks")
        if not os.path.exists(tasks_dir):
            raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

        # Scan and filter traces
        task_dirs = sorted(
            d
            for d in os.listdir(tasks_dir)
            if os.path.isdir(os.path.join(tasks_dir, d))
        )

        # Phase 1: Load and filter traces (sync I/O)
        filtered: List[Tuple[str, Dict[str, Any], List[str]]] = []  # (task_id, trace, image_paths)
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

            input_images = trace.get("input", {}).get("images", [])
            image_paths = [
                img["path"] if isinstance(img, dict) else img
                for img in input_images
            ]
            filtered.append((task_id, trace, image_paths))

        # Phase 2: Generate captions concurrently (if api_pool provided)
        if api_pool:
            caption_concurrency = 8
            sem = asyncio.Semaphore(caption_concurrency)

            async def _cap(paths: List[str]) -> str:
                if not paths:
                    return ""
                async with sem:
                    return await cls._generate_caption(api_pool, paths)

            caption_tasks = [_cap(paths) for _, _, paths in filtered]
            caption_results = await asyncio.gather(*caption_tasks, return_exceptions=True)
            captions_list: List[str] = []
            for i, result in enumerate(caption_results):
                if isinstance(result, Exception):
                    logger.warning(f"Caption failed for {filtered[i][0]}: {result}")
                    captions_list.append("")
                else:
                    captions_list.append(result)
                    if result:
                        logger.info(f"  {filtered[i][0]}: caption='{result[:60]}...'")
        else:
            captions_list = [""] * len(filtered)

        # Phase 3: Build index texts
        task_ids: List[str] = []
        texts: List[str] = []
        captions: List[str] = []

        for idx, (task_id, trace, _) in enumerate(filtered):
            caption = captions_list[idx]
            index_text = cls.build_index_text(trace, caption=caption)
            if not index_text.strip():
                continue

            task_ids.append(task_id)
            texts.append(index_text)
            captions.append(caption)

        if not task_ids:
            raise ValueError(
                f"No valid traces found in {tasks_dir} "
                f"(filter_correct={filter_correct})"
            )

        logger.info(f"Building trace bank from {len(task_ids)} traces ...")

        # Embed
        embeddings = await embedder.encode_batch(texts, batch_size=batch_size)

        # Save
        bank_dir = os.path.join(memory_dir, bank_dir_name)
        os.makedirs(bank_dir, exist_ok=True)

        np.save(os.path.join(bank_dir, "embeddings.npy"), embeddings)
        with open(os.path.join(bank_dir, "task_ids.json"), "w", encoding="utf-8") as f:
            json.dump(task_ids, f, ensure_ascii=False)
        with open(os.path.join(bank_dir, "captions.json"), "w", encoding="utf-8") as f:
            json.dump(captions, f, ensure_ascii=False)

        logger.info(
            f"Trace bank saved to {bank_dir}: "
            f"{len(task_ids)} entries, dim={embeddings.shape[1]}"
        )

        return cls(memory_dir, bank_dir_name=bank_dir_name)
