"""Trace-level retrieval pipeline orchestrator.

Coordinates query rewriting, embedding search, reranking, LLM summary,
and sufficiency checking to produce experience for the agent.
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
You are summarizing experience from similar successfully solved tasks. \
This experience will guide the solving of similar tasks in the future.

## Current Task
Question: {question}
{image_line}
## Similar Solved Examples
{examples}

Write concise sentences of actionable experience. Focus on:
- What general approach or tool usage strategy worked well
- What conditions or situations require special attention
- What common mistakes to avoid

Write in a direct, advisory tone as if briefing a colleague. Do not reference \
specific example numbers or task labels. Do not repeat the question. \
Do not include any specific answers, answer choices, or concrete analysis of \
particular examples. The experience should be universally applicable to similar tasks."""

SUFFICIENCY_PROMPT = """\
Judge whether the following experience is sufficient to guide solving the \
given task. Consider both the question and the visual content of the task.

Task: {question}
{image_line}
Experience:
{summary}

Respond in JSON: {{"sufficient": true/false, "missing": "if not sufficient, \
one phrase describing the unaddressed aspect; empty string if sufficient"}}"""


class TracePipeline:
    """Trace-level retrieval: query rewrite -> embed -> search -> rerank -> summary."""

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

    async def close(self) -> None:
        """Release resources (HTTP clients, etc.)."""
        if self.reranker and hasattr(self.reranker, "close"):
            await self.reranker.close()

    async def run(
        self, question: str, images: Optional[List[str]] = None
    ) -> str:
        """Run the full retrieval pipeline.

        Flow per round:
            rewrite -> embed (batch) -> search -> dedup -> rerank -> summary -> sufficiency

        Args:
            question: User's question
            images: Input image paths

        Returns:
            Experience string.  Empty string if nothing useful found.
        """
        assert self.config.max_retrieval_rounds >= 1, (
            f"max_retrieval_rounds must be >= 1, got {self.config.max_retrieval_rounds}"
        )

        search_context = ""
        image_caption = ""
        summary = ""

        for round_idx in range(self.config.max_retrieval_rounds):
            logger.info(f"Retrieval round {round_idx + 1}/{self.config.max_retrieval_rounds}")

            # --- Step 1: Query Rewrite ---
            if self.query_rewriter and self.config.enable_query_rewrite:
                query_result = await self.query_rewriter.rewrite(
                    question,
                    images,
                    previous_context=search_context,
                )
                text_queries = query_result["text_queries"]
                # Capture image_caption from first round (rewriter sees the images)
                if round_idx == 0:
                    image_caption = query_result.get("image_caption", "")
                logger.info(f"  Rewrite → {len(text_queries)} queries, caption={bool(image_caption)}")
            else:
                text_queries = [question]

            # --- Step 2: Embed (batch) + Search ---
            # Fresh candidates each round (new queries target the gap)
            candidates: List[Dict[str, Any]] = []
            all_embs = await self.embedder.encode_text(text_queries)
            for i in range(len(text_queries)):
                results = self.memory_bank.search(
                    all_embs[i: i + 1],
                    top_k=self.config.retrieval_top_k,
                    min_score=self.config.min_score,
                )
                candidates.extend(results)

            # Deduplicate by task_id
            before_dedup = len(candidates)
            candidates = self._deduplicate(candidates)
            logger.info(
                f"  Search → {before_dedup} hits, {len(candidates)} after dedup "
                f"(min_score={self.config.min_score})"
            )

            # No candidates above threshold — skip to next round or finish
            if not candidates:
                logger.info("  No candidates above min_score, skipping to next round")
                continue

            # --- Step 3: Rerank ---
            rerank_query = question
            if image_caption:
                rerank_query = f"{question}\nImage description: {image_caption}"

            if self.config.enable_rerank and self.reranker:
                for c in candidates:
                    if "rerank_text" not in c:
                        c["rerank_text"] = MemoryBank.build_index_text(
                            c, caption=c.get("_caption", "")
                        )

                candidates = await self.reranker.rerank(
                    rerank_query,
                    candidates,
                    top_n=self.config.rerank_top_n,
                    text_key="rerank_text",
                )
                logger.info(
                    f"  Rerank → top {len(candidates)}, "
                    f"scores: {[round(c.get('rerank_score', 0), 4) for c in candidates]}"
                )
            else:
                candidates.sort(
                    key=lambda x: x.get("retrieval_score", 0), reverse=True
                )
                candidates = candidates[: self.config.rerank_top_n]

            # --- Step 4: Summary ---
            summary = await self._summarize(question, image_caption, candidates)
            logger.info(f"  Summary → {len(summary)} chars")

            # --- Step 5: Sufficiency Check (multi-round only) ---
            if (
                self.config.max_retrieval_rounds > 1
                and round_idx < self.config.max_retrieval_rounds - 1
                and self.api_pool
            ):
                is_sufficient, missing = await self._judge_sufficiency(
                    question, image_caption, summary
                )
                logger.info(f"  Sufficiency → sufficient={is_sufficient}, missing='{missing}'")
                if is_sufficient:
                    return summary
                # Pass summary + gap to next round's rewriter
                search_context = f"Current experience:\n{summary}\n\nGap: {missing}"
            else:
                return summary

        # Exhausted all rounds — return whatever we have
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _summarize(
        self,
        question: str,
        image_caption: str,
        candidates: List[Dict[str, Any]],
    ) -> str:
        """Summarize top candidates into experience via LLM."""
        image_line = f"Image description: {image_caption}" if image_caption else ""
        examples_text = self._format_candidates(candidates)
        prompt = SUMMARY_PROMPT.format(
            question=question, image_line=image_line, examples=examples_text
        )

        try:
            result = await self.api_pool.execute(
                "qa",
                system="You are a concise experience summarizer.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400,
            )
            return result.get("answer", "")
        except Exception as e:
            logger.warning(f"Summary failed: {e}")
            # Fallback: list tools from each candidate
            seen_tools: set = set()
            for c in candidates:
                for s in c.get("trace", []):
                    if s.get("type") == "action" and "tool" in s:
                        seen_tools.add(s["tool"])
            return f"Similar tasks used tools: {', '.join(seen_tools)}." if seen_tools else ""

    async def _judge_sufficiency(
        self, question: str, image_caption: str, summary: str
    ) -> tuple:
        """Judge if the summarized experience is sufficient for the task.

        Returns:
            Tuple of (is_sufficient: bool, missing: str)
        """
        image_line = f"Image description: {image_caption}" if image_caption else ""
        prompt = SUFFICIENCY_PROMPT.format(
            question=question, image_line=image_line, summary=summary
        )

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
                return parsed.get("sufficient", True), parsed.get("missing", "")
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
        """Format candidates for the summary LLM prompt.

        Each candidate shows: image description, question, and the full
        reasoning chain (think/answer content + [tool_name], observations skipped).
        """
        parts = []
        for i, c in enumerate(candidates, 1):
            question = c.get("input", {}).get("question", "N/A")
            caption = c.get("_caption", "")

            # Build reasoning chain: think/answer → content, action → [tool], skip observation
            steps = []
            for s in c.get("trace", []):
                stype = s.get("type", "")
                if stype in ("think", "answer") and s.get("content"):
                    steps.append(s["content"])
                elif stype == "action" and "tool" in s:
                    steps.append(f"[{s['tool']}]")
                # observation → skip

            steps_str = "\n  ".join(
                f"{j}. {step}" for j, step in enumerate(steps, 1)
            ) if steps else "N/A"

            lines = [f"### Example {i}"]
            if caption:
                lines.append(f"- Image description: {caption}")
            lines.append(f"- Question: {question}")
            lines.append(f"- Steps:\n  {steps_str}")

            parts.append("\n".join(lines))
        return "\n\n".join(parts)
