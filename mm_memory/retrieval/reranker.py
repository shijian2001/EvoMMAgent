"""Reranker client — calls vLLM /v1/rerank endpoint (Qwen3-VL-Reranker)."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class Reranker:
    """Reranking via vLLM /v1/rerank with a persistent HTTP client."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str = "dummy",
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = timeout

    async def _get_client(self) -> httpx.AsyncClient:
        """Return a persistent async HTTP client, creating one if needed."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def rerank(
        self, query: str, candidates: List[Dict[str, Any]],
        top_n: int = 3, text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """Rerank candidates. Returns top-N sorted by score desc, each with ``rerank_score``."""
        texts = [c[text_key] for c in candidates]
        client = await self._get_client()

        for attempt in range(self.max_retries):
            try:
                resp = await client.post(
                    f"{self.base_url}/rerank",
                    json={"model": self.model_name, "query": query, "documents": texts},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                resp.raise_for_status()

                scored = []
                for item in resp.json()["results"]:
                    c = candidates[item["index"]].copy()
                    c["rerank_score"] = item["relevance_score"]
                    scored.append(c)

                scored.sort(key=lambda x: x["rerank_score"], reverse=True)
                return scored[:top_n]

            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = 0.5 * (1.5 ** attempt)
                    logger.warning(
                        f"Reranker failed ({attempt + 1}/{self.max_retries}): {e}, "
                        f"retry in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        f"Rerank failed after {self.max_retries} attempts: {e}. Falling back."
                    )
                    return candidates[:top_n]

        return candidates[:top_n]
