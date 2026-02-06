"""Query rewriter using the agent's shared LLM via APIPool + JSONParser.

Generates alternative text queries to improve retrieval recall.
Currently implements ``text_only`` strategy; ``auto`` strategy
(with image-based queries) is reserved for future extension.
"""

import logging
from typing import Any, Dict, List, Optional

from api.json_parser import JSONParser

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """You are a query rewriter for a multimodal retrieval system.

Given a user question, generate {max_queries} alternative search queries that \
capture different aspects of the question. These queries will be used to retrieve \
similar solved examples from a memory bank.

Rules:
- Query 1: Rephrase the question more generally (focus on the task type)
- Query 2: Focus on the specific visual analysis needed
- Query 3: Focus on what tools/methods might help solve this
- Keep each query concise (under 30 words)

User question: {question}
{context_block}
Output as JSON: {{"queries": ["query1", "query2", ...]}}"""


class QueryRewriter:
    """Query rewriting via shared APIPool + JSONParser."""

    def __init__(self, api_pool, max_sub_queries: int = 3):
        """Initialize query rewriter.

        Args:
            api_pool: Shared APIPool instance from the agent
            max_sub_queries: Maximum number of rewritten queries to generate
        """
        self.api_pool = api_pool
        self.max_sub_queries = max_sub_queries

    async def rewrite(
        self,
        question: str,
        images: Optional[List[str]] = None,
        strategy: str = "text_only",
        previous_context: str = "",
    ) -> Dict[str, Any]:
        """Rewrite query for retrieval.

        Args:
            question: Original question text
            images: Image paths (reserved for future ``auto`` strategy)
            strategy: ``"text_only"`` or ``"auto"`` (future)
            previous_context: Context from previous retrieval round (for multi-round)

        Returns:
            Dict with keys:
                - ``text_queries``: ``[original_question, rewritten_1, ...]``
                - ``image_queries``: ``[]`` (reserved)
                - ``multimodal_queries``: ``[]`` (reserved)
        """
        if strategy == "text_only":
            queries = await self._text_only_rewrite(question, previous_context)
        else:
            # Future: "auto" strategy with image analysis
            queries = await self._text_only_rewrite(question, previous_context)

        return {
            "text_queries": [question] + queries,  # Always include original
            "image_queries": [],
            "multimodal_queries": [],
        }

    async def _text_only_rewrite(
        self, question: str, previous_context: str = ""
    ) -> List[str]:
        """Generate text-only rewritten queries via LLM."""
        context_block = ""
        if previous_context:
            context_block = (
                f"\nPrevious retrieval context (use this to refine your queries):\n"
                f"{previous_context}\n"
            )

        prompt = REWRITE_PROMPT.format(
            question=question,
            max_queries=self.max_sub_queries,
            context_block=context_block,
        )

        try:
            result = await self.api_pool.execute(
                "qa",
                system="You are a helpful query rewriter. Always respond in valid JSON.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )

            answer = result.get("answer", "")
            parsed = JSONParser.parse(answer)

            if isinstance(parsed, dict) and "queries" in parsed:
                return parsed["queries"][: self.max_sub_queries]

            return []

        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}, using original query only")
            return []
