"""State-level retrieval pipeline.

Embeds the current agent state, searches the StateBank for similar
decision points, and returns pre-computed experience strings.
No online LLM calls, no reranking — all experience is generated offline.
"""

import logging

from mm_memory.state_bank import StateBank

logger = logging.getLogger(__name__)


class StatePipeline:
    """State-level retrieval: embed → cosine search → return top experience."""

    def __init__(self, config, state_bank: StateBank, embedder):
        """Initialize the state retrieval pipeline.

        Args:
            config: ``RetrievalConfig`` instance
            state_bank: Pre-loaded ``StateBank``
            embedder: ``Embedder`` instance for query encoding
        """
        self.config = config
        self.state_bank = state_bank
        self.embedder = embedder

    async def retrieve(self, state_text: str) -> str:
        """Retrieve experience for the current state.

        Args:
            state_text: Serialized state text from ``StateBank.state_to_text()``

        Returns:
            Concatenated experience string, or empty string if nothing found.
        """
        # Embed current state
        emb = await self.embedder.encode_text([state_text])

        # Cosine search with Q-value filtering
        candidates = self.state_bank.search(
            emb,
            top_k=self.config.retrieval_top_k,
            min_score=self.config.min_score,
            min_q=self.config.min_q_value,
        )

        if not candidates:
            return ""

        # Take top-n and format with source labels
        entries = []
        for i, c in enumerate(candidates[: self.config.experience_top_n], 1):
            exp = c.get("experience", "")
            if not exp:
                continue
            source = c.get("source", "correct")
            tag = "Learned from success" if source == "correct" else "Learned from failure"
            entries.append(f"#{i} [{tag}]\n{exp}")

        if not entries:
            return ""

        if len(entries) == 1:
            preamble = (
                "The following experience is retrieved from a similar reasoning state.\n"
                "Use your own judgment to decide whether to follow it."
            )
        else:
            preamble = (
                f"The following {len(entries)} experiences are retrieved from similar "
                f"reasoning states, ranked by relevance.\n"
                "Use your own judgment to decide how to incorporate them."
            )

        return preamble + "\n\n" + "\n\n".join(entries)

    async def close(self) -> None:
        """Release resources (no-op for state pipeline)."""
        pass
