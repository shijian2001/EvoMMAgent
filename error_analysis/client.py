"""Async LLM client with concurrency control and retry."""

import asyncio
import base64
import logging
import mimetypes
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_API_URL = (
    "https://runway.devops.rednote.life/openai/chat/completions"
    "?api-version=2024-12-01-preview"
)


class AsyncLLMClient:
    """Async LLM client gated by semaphore with exponential-backoff retry."""

    def __init__(
        self,
        api_key: str,
        api_url: str = DEFAULT_API_URL,
        concurrency: int = 10,
        max_retries: int = 3,
        timeout: int = 120,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(concurrency)
        self._timeout = httpx.Timeout(timeout)
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            headers={"api-key": self.api_key, "Content-Type": "application/json"},
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.aclose()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    async def call(self, prompt: str, image_paths: Optional[List[str]] = None) -> str:
        """Send prompt with optional images, return LLM response text."""
        content: list = [{"type": "text", "text": prompt}]
        for p in image_paths or []:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._encode_image(p)},
            })
        payload = {"messages": [{"role": "user", "content": content}]}

        async with self._semaphore:
            return await self._request(payload)

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _encode_image(path: str) -> str:
        """Read local file and return base64 data-URL."""
        mime, _ = mimetypes.guess_type(path)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:{mime or 'image/jpeg'};base64,{b64}"

    async def _request(self, payload: dict) -> str:
        """POST with exponential-backoff retry."""
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = await self._client.post(self.api_url, json=payload)  # type: ignore[union-attr]
                if resp.status_code != 200:
                    logger.error("HTTP %d response body: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    delay = min(0.5 * (1.5 ** attempt), 3.0)
                    logger.warning(
                        "Retry %d/%d: %s — wait %.1fs",
                        attempt + 1, self.max_retries, e, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("Failed after %d retries: %s", self.max_retries, e)
        raise RuntimeError(f"All {self.max_retries} retries exhausted") from last_err
