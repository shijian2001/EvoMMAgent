"""Embedding client for vLLM-deployed embedding models.

Uses AsyncOpenAI to call the standard /v1/embeddings endpoint
served by vLLM.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

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

    async def encode_multimodal(self, text: str, image_paths: List[str]) -> np.ndarray:
        """Encode one multimodal query (text + one or more images).

        For multiple images, embeddings are averaged across per-image encodings.
        """
        valid_paths = [p for p in image_paths if p and os.path.exists(p)]
        if not valid_paths:
            return await self.encode_text([text or ""])

        per_image_embeddings: List[np.ndarray] = []
        for path in valid_paths:
            image_uri = f"file://{os.path.abspath(path)}"
            payload = {"text": text or "", "image": image_uri}

            for attempt in range(self.max_retries):
                try:
                    response = await self.client.embeddings.create(
                        model=self.model_name,
                        input=[payload],
                    )
                    emb = np.array([response.data[0].embedding], dtype=np.float32)
                    self._dim = emb.shape[1]
                    per_image_embeddings.append(emb)
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        delay = 0.5 * (1.5 ** attempt)
                        logger.warning(
                            f"Embedder encode_multimodal failed (attempt {attempt + 1}/{self.max_retries}): "
                            f"{e} - Retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise

        if not per_image_embeddings:
            return await self.encode_text([text or ""])

        stacked = np.vstack(per_image_embeddings)
        return np.mean(stacked, axis=0, keepdims=True).astype(np.float32)

    async def encode_view(self, composed: Dict[str, Any]) -> np.ndarray:
        """Encode one composed view payload."""
        text = str(composed.get("text", ""))
        images = composed.get("images") or []
        if images:
            return await self.encode_multimodal(text, images)
        return await self.encode_text([text])

    async def encode_view_batch(
        self, composed_list: List[Dict[str, Any]], batch_size: int = 32
    ) -> np.ndarray:
        """Encode a list of composed view payloads in order."""
        if not composed_list:
            return np.zeros((0, self.dim or 0), dtype=np.float32)

        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(composed_list), batch_size):
            batch = composed_list[i : i + batch_size]
            batch_embs = await asyncio.gather(*[self.encode_view(item) for item in batch])
            all_embeddings.append(np.vstack(batch_embs))
            logger.info(
                f"Embedded {min(i + batch_size, len(composed_list))}/{len(composed_list)} composed views"
            )
        return np.vstack(all_embeddings)
