"""Retrieval pipeline orchestrator.

Coordinates query rewriting, embedding search, reranking, sufficiency
checking, and LLM summary to produce experience for the agent.
"""

import logging
from typing import Any, Dict, List, Optional

from api.json_parser import JSONParser
from mm_memory.memory_bank import MemoryBank

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """\
Based on the following retrieved examples that are similar to the current task, \
summarize the experience.

## Current Task
Question: {question}

## Retrieved Similar Examples
{examples}

Summarize the experience in 2-3 sentences:
1. What tools/approach are most likely effective?
2. Any key patterns or pitfalls from similar tasks?

Keep it concise and actionable."""

SUFFICIENCY_PROMPT = """\
Given the current task and retrieved examples, do you have enough information \
to provide useful experience?

Question: {question}
Retrieved examples summary: {summary}

Respond in JSON: {{"sufficient": true/false, "reason": "brief explanation if not sufficient"}}"""


class RetrievalPipeline:
    """Orchestrates query rewrite -> embed -> search -> rerank -> summary."""

    def __init__(
        self,
        config,
        memory_bank: MemoryBank,
        embedder,
        reranker=None,
        api_pool=None,
        query_rewriter=None,
    ):
        """Initialize the retrieval pipeline.

        Args:
            config: ``RetrievalConfig`` instance
            memory_bank: Pre-loaded ``MemoryBank``
            embedder: ``Embedder`` instance for query encoding
            reranker: Optional ``Reranker`` (None if ``enable_rerank=False``)
            api_pool: Agent's shared ``APIPool`` (for summary & sufficiency calls)
            query_rewriter: Optional ``QueryRewriter`` (None if disabled)
        """
        self.config = config
        self.memory_bank = memory_bank
        self.embedder = embedder
        self.reranker = reranker
        self.api_pool = api_pool
        self.query_rewriter = query_rewriter

    async def run(
        self, question: str, images: Optional[List[str]] = None
    ) -> str:
        """Run the full retrieval pipeline.

        Args:
            question: User's question
            images: Input image paths (reserved for future use)

        Returns:
            Experience string.  Empty string if nothing useful found.
        """
        all_candidates: List[Dict[str, Any]] = []
        search_context = ""

        for round_idx in range(self.config.max_retrieval_rounds):
            # --- Step 1: Query Rewrite ---
            if self.query_rewriter and self.config.enable_query_rewrite:
                query_result = await self.query_rewriter.rewrite(
                    question,
                    images,
                    strategy=self.config.query_rewrite_strategy,
                    previous_context=search_context,
                )
                text_queries = query_result["text_queries"]
            else:
                text_queries = [question]

            # --- Step 2: Embed + Search ---
            for q in text_queries:
                q_emb = await self.embedder.encode_text([q])
                results = self.memory_bank.search(
                    q_emb, top_k=self.config.retrieval_top_k
                )
                all_candidates.extend(results)

            # Deduplicate by task_id
            all_candidates = self._deduplicate(all_candidates)

            # --- Step 3: Rerank ---
            if self.config.enable_rerank and self.reranker:
                # Build retrieval text for reranker scoring
                for c in all_candidates:
                    if "rerank_text" not in c:
                        c["rerank_text"] = MemoryBank.build_index_text(c)

                all_candidates = await self.reranker.rerank(
                    question,
                    all_candidates,
                    top_n=self.config.rerank_top_n,
                    text_key="rerank_text",
                )
            else:
                # Sort by retrieval score, take top_n
                all_candidates.sort(
                    key=lambda x: x.get("retrieval_score", 0), reverse=True
                )
                all_candidates = all_candidates[: self.config.rerank_top_n]

            # --- Step 4: Sufficiency Check (multi-round only) ---
            if (
                self.config.max_retrieval_rounds > 1
                and round_idx < self.config.max_retrieval_rounds - 1
                and self.api_pool
            ):
                is_sufficient, reason = await self._judge_sufficiency(
                    question, all_candidates
                )
                if is_sufficient:
                    break
                search_context = self._build_search_context(all_candidates, reason)

        # --- Step 5: Summary (always) ---
        if not all_candidates:
            return ""

        return await self._summarize(question, all_candidates)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _summarize(
        self, question: str, candidates: List[Dict[str, Any]]
    ) -> str:
        """Summarize top candidates into experience via LLM."""
        examples_text = self._format_candidates(candidates)
        prompt = SUMMARY_PROMPT.format(question=question, examples=examples_text)

        try:
            result = await self.api_pool.execute(
                "qa",
                system="You are a concise experience summarizer.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300,
            )
            return result.get("answer", "")
        except Exception as e:
            logger.warning(f"Summary failed: {e}")
            # Fallback: use trace info directly
            return "\n".join(
                f"- Similar task: {c.get('input', {}).get('question', '')}, "
                f"tools: {', '.join(s['tool'] for s in c.get('trace', []) if s.get('type') == 'action')}"
                for c in candidates
            )

    async def _judge_sufficiency(
        self, question: str, candidates: List[Dict[str, Any]]
    ) -> tuple:
        """Judge if retrieved info is sufficient.

        Returns:
            Tuple of (is_sufficient: bool, reason: str)
        """
        summary = self._brief_summary(candidates)
        prompt = SUFFICIENCY_PROMPT.format(question=question, summary=summary)

        try:
            result = await self.api_pool.execute(
                "qa",
                system="Respond in valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            parsed = JSONParser.parse(result.get("answer", ""))
            if isinstance(parsed, dict):
                return parsed.get("sufficient", True), parsed.get("reason", "")
            return True, ""
        except Exception:
            return True, ""  # Default: sufficient (don't loop on error)

    def _deduplicate(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate by task_id, keeping the highest score."""
        seen: Dict[str, Dict[str, Any]] = {}
        for c in candidates:
            tid = c.get("task_id", id(c))
            existing_score = seen.get(tid, {}).get("retrieval_score", -1)
            if tid not in seen or c.get("retrieval_score", 0) > existing_score:
                seen[tid] = c
        return list(seen.values())

    def _format_candidates(self, candidates: List[Dict[str, Any]]) -> str:
        """Format candidates for the summary LLM prompt."""
        parts = []
        for i, c in enumerate(candidates, 1):
            question = c.get("input", {}).get("question", "N/A")
            answer = c.get("answer", "N/A")
            tools = [
                s["tool"]
                for s in c.get("trace", [])
                if s.get("type") == "action"
            ]
            tools_str = ", ".join(tools) if tools else "none"

            # Include think steps for richer context
            thinks = [
                s["content"]
                for s in c.get("trace", [])
                if s.get("type") == "think" and s.get("content")
            ]
            think_str = " | ".join(thinks[:2]) if thinks else "N/A"

            parts.append(
                f"### Example {i}\n"
                f"- Question: {question}\n"
                f"- Answer: {answer}\n"
                f"- Tools used: {tools_str}\n"
                f"- Reasoning: {think_str}"
            )
        return "\n\n".join(parts)

    def _brief_summary(self, candidates: List[Dict[str, Any]]) -> str:
        """Brief summary for sufficiency check."""
        return "; ".join(
            f"[{c.get('sub_task', '')}] "
            f"{c.get('input', {}).get('question', '')[:80]}"
            for c in candidates
        )

    def _build_search_context(
        self, candidates: List[Dict[str, Any]], reason: str
    ) -> str:
        """Build context for next round query rewrite."""
        ctx = f"Previous retrieval found: {self._brief_summary(candidates)}\n"
        ctx += f"Still need: {reason}"
        return ctx
