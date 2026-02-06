"""Reranker client for vLLM-deployed reranker models.

Uses httpx to call the /v1/score endpoint served by vLLM,
which is not part of the standard OpenAI API.
"""

import asyncio
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class Reranker:
    """Reranking via vLLM /v1/score endpoint."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str = "dummy",
        max_retries: int = 3,
    ):
        """Initialize reranker client.

        Args:
            model_name: Name of the reranker model deployed on vLLM
            base_url: vLLM endpoint (e.g. "http://host:8002/v1")
            api_key: API key (use "dummy" for local vLLM)
            max_retries: Maximum retry attempts with exponential backoff
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 3,
        text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """Rerank candidates by relevance to query.

        Args:
            query: Search query text
            candidates: List of dicts, each must have a ``text_key`` field
            top_n: Number of top results to return
            text_key: Key in candidate dict containing the scoring text

        Returns:
            Top-N candidates sorted by score descending,
            each with an added ``rerank_score`` field.
            On failure, falls back to returning candidates[:top_n].
        """
        import httpx

        texts = [c[text_key] for c in candidates]

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self.base_url}/score",
                        json={
                            "model": self.model_name,
                            "text_1": query,
                            "text_2": texts,
                        },
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                    response.raise_for_status()

                data = response.json()["data"]

                scored: List[Dict[str, Any]] = []
                for item in data:
                    idx = item["index"]
                    candidate = candidates[idx].copy()
                    candidate["rerank_score"] = item["score"]
                    scored.append(candidate)

                scored.sort(key=lambda x: x["rerank_score"], reverse=True)
                return scored[:top_n]

            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = 0.5 * (1.5 ** attempt)
                    logger.warning(
                        f"Reranker failed (attempt {attempt + 1}/{self.max_retries}): "
                        f"{e} - Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        f"Rerank failed after {self.max_retries} attempts: {e}. "
                        f"Falling back to top-{top_n} without reranking."
                    )
                    return candidates[:top_n]

        # Should not reach here, but satisfy type checker
        return candidates[:top_n]
