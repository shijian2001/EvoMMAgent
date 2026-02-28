"""Trace-level retrieval pipeline: multimodal embed -> cosine top-1 search."""

import logging
from typing import List, Optional

from mm_memory.trace_bank import TraceBank

logger = logging.getLogger(__name__)


class TracePipeline:
    """Minimal trace-level retrieval with precomputed experiences."""

    def __init__(self, config, trace_bank: TraceBank, embedder):
        self.config = config
        self.trace_bank = trace_bank
        self.embedder = embedder

    async def close(self) -> None:
        """No-op for interface compatibility."""
        return

    async def run(
        self,
        question: str,
        images: Optional[List[str]] = None,
        sub_task: str = "",
    ) -> str:
        """Retrieve top-1 experience using multimodal embedding."""
        query_parts = [f"Question: {question}"]
        if sub_task:
            query_parts.append(f"Task: {sub_task}")
        query_text = "\n".join(query_parts)

        query_emb = await self.embedder.encode_multimodal(query_text, images)

        experience = self.trace_bank.search(
            query_emb,
            min_score=self.config.min_score,
        )
        return experience or ""
