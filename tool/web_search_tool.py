"""Web search tool via Serper.dev API (text search only)."""

import os
import json
import time
import logging
import asyncio
import requests
from typing import Dict, Union, Optional

from dotenv import load_dotenv
from tool.base_tool import BasicTool, register_tool

for _env_path in ("api/utils/keys.env", "keys.env", ".env"):
    if os.path.exists(_env_path):
        load_dotenv(_env_path)

GOOGLE_SEARCH_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY", "")


@register_tool(name="web_search")
class WebSearchTool(BasicTool):
    """Web search tool for text queries via Serper.dev (Google)."""

    name = "web_search"
    description_en = (
        "Search the web for information using a text query. "
        "Returns the top-k results, each with a title and snippet."
    )
    description_zh = (
        "使用文本查询搜索网页信息。"
        "返回前 k 条结果，每条包含标题和摘要。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (1-5, default: 3)",
            },
        },
        "required": ["query"],
    }
    example = '{"query": "Which city is the capital of China?", "top_k": 3}'

    def __init__(self, cfg: Optional[Dict] = None, use_zh: bool = False):
        super().__init__(cfg, use_zh)
        self._max_retries = 3
        self._retry_delay = 2

    def call(self, params: Union[str, Dict]) -> Dict:
        params_dict = self.parse_params(params)
        query = params_dict.get("query", "").strip()
        top_k = max(1, min(int(params_dict.get("top_k", 3)), 5))

        if not query:
            return {"error": "Parameter 'query' is required"}

        try:
            return self._text_search(query, top_k)
        except Exception as e:
            logging.error(f"Web search failed: {e}")
            return {"error": f"Search failed: {str(e)}"}

    async def call_async(self, params: Union[str, Dict]) -> Dict:
        return await asyncio.to_thread(self.call, params)

    def _text_search(self, query: str, top_k: int) -> Dict:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": GOOGLE_SEARCH_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "gl": "cn",
            "hl": "zh-cn",
            "location": "China",
            "num": top_k,
        }

        for attempt in range(self._max_retries):
            try:
                resp = requests.post(
                    url, headers=headers, data=json.dumps(payload), timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("organic", [])[:top_k]

                if not results:
                    return {"search results": "No results found"}

                lines = []
                for i, res in enumerate(results, 1):
                    title = res.get("title", "")
                    snippet = res.get("snippet", "")
                    lines.append(f"[{i}] Title: {title}")
                    lines.append(f"Snippet: {snippet}")
                    lines.append("")

                return {"search results": "\n".join(lines).strip()}
            except Exception as e:
                logging.warning(
                    f"Request failed (attempt {attempt + 1}/{self._max_retries}): {e}"
                )
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay)
                else:
                    raise
