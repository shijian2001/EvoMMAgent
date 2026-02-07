"""Query rewriter using the agent's shared LLM via APIPool + JSONParser.

Generates an image caption (when images are provided) and variable-count
alternative text queries to improve retrieval recall.  The LLM sees both
the question and all input images in a single multimodal call.
"""

import logging
from typing import Any, Dict, List, Optional

from api.json_parser import JSONParser

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """\
You are a query rewriter for a multimodal retrieval system.
Your goal: generate search queries to retrieve similar solved examples from a memory bank.

Given the user's question{and_images}, output JSON with:
1. "image_caption": A concise 1-2 sentence description of what the images show as a whole. \
If no images are provided, set to "".
2. "queries": A list of 0 to {max_queries} alternative search queries that capture different \
retrieval angles (task type, visual analysis needed, tools/methods that might help). \
Generate only as many as useful — do not pad with redundant queries.

Keep each query and the caption under 30 words. Be specific, not generic.

User question: {question}
{context_block}
Output as JSON: {{"image_caption": "...", "queries": [...]}}"""


class QueryRewriter:
    """Multimodal query rewriting via shared APIPool + JSONParser."""

    def __init__(self, api_pool, max_sub_queries: int = 5):
        """Initialize query rewriter.

        Args:
            api_pool: Shared APIPool instance from the agent
            max_sub_queries: Maximum number of additional rewritten queries
        """
        self.api_pool = api_pool
        self.max_sub_queries = max_sub_queries

    async def rewrite(
        self,
        question: str,
        images: Optional[List[str]] = None,
        previous_context: str = "",
    ) -> Dict[str, Any]:
        """Rewrite query for retrieval.

        Sends the question (and images, if any) to the LLM in a single
        multimodal call.  Returns the original question, an optional
        image caption, and 0-N additional search queries.

        Args:
            question: Original question text
            images: Optional list of input image paths
            previous_context: Context from previous retrieval round (multi-round)

        Returns:
            Dict with keys:
                - ``text_queries``: ``[question, caption?, q1, q2, ...]``
                - ``image_caption``: caption string (empty if no images)
        """
        caption, extra_queries = await self._rewrite_impl(
            question, images, previous_context
        )

        text_queries = [question]
        if caption:
            text_queries.append(caption)
        text_queries.extend(extra_queries[: self.max_sub_queries])

        return {"text_queries": text_queries, "image_caption": caption}

    # ------------------------------------------------------------------

    async def _rewrite_impl(
        self,
        question: str,
        images: Optional[List[str]] = None,
        previous_context: str = "",
    ) -> tuple:
        """Core implementation: call LLM and parse output.

        Returns:
            (image_caption: str, queries: List[str])
        """
        context_block = ""
        if previous_context:
            context_block = (
                f"\nPrevious retrieval context (use this to refine your queries):\n"
                f"{previous_context}\n"
            )

        and_images = " and images" if images else ""
        prompt = REWRITE_PROMPT.format(
            question=question,
            max_queries=self.max_sub_queries,
            context_block=context_block,
            and_images=and_images,
        )

        # Build multimodal message when images are provided
        if images:
            content: Any = [{"type": "text", "text": prompt}]
            for img_path in images:
                content.append(
                    {"type": "image_url", "image_url": {"url": img_path}}
                )
        else:
            content = prompt

        try:
            result = await self.api_pool.execute(
                "qa",
                system="You are a helpful query rewriter. Always respond in valid JSON.",
                messages=[{"role": "user", "content": content}],
                temperature=0.3,
                max_tokens=500,
            )

            answer = result.get("answer", "")
            parsed = JSONParser.parse(answer)

            if isinstance(parsed, dict):
                caption = parsed.get("image_caption", "") or ""
                queries = parsed.get("queries", []) or []
                return caption, queries

            return "", []

        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}, using original query only")
            return "", []
