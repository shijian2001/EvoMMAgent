"""Embedding client for vLLM-deployed embedding models.

Uses AsyncOpenAI to call the standard /v1/embeddings endpoint
served by vLLM.
"""

import asyncio
import logging
from typing import List, Optional

import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class Embedder:
    """Embedding via vLLM /v1/embeddings endpoint."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str = "dummy",
        max_retries: int = 3,
    ):
        """Initialize embedding client.

        Args:
            model_name: Name of the embedding model deployed on vLLM
            base_url: vLLM endpoint (e.g. "http://host:8001/v1")
            api_key: API key (use "dummy" for local vLLM)
            max_retries: Maximum retry attempts with exponential backoff
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._dim: Optional[int] = None

    @property
    def dim(self) -> Optional[int]:
        """Embedding dimension (available after first encode call)."""
        return self._dim

    async def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into embeddings.

        Args:
            texts: List of text strings

        Returns:
            np.ndarray of shape [N, D], dtype float32
        """
        for attempt in range(self.max_retries):
            try:
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                )
                embeddings = [item.embedding for item in response.data]
                result = np.array(embeddings, dtype=np.float32)
                self._dim = result.shape[1]
                return result
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = 0.5 * (1.5 ** attempt)
                    logger.warning(
                        f"Embedder encode_text failed (attempt {attempt + 1}/{self.max_retries}): "
                        f"{e} - Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

    async def encode_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> np.ndarray:
        """Encode a large list of texts in batches.

        Useful for offline memory bank construction.

        Args:
            texts: List of text strings
            batch_size: Number of texts per API call

        Returns:
            np.ndarray of shape [N, D], dtype float32
        """
        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = await self.encode_text(batch)
            all_embeddings.append(emb)
            logger.info(
                f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} texts"
            )
        return np.vstack(all_embeddings)
