"""State-level retrieval helper (single-view wrapper for compatibility)."""

from typing import Any, Dict, List

from mm_memory.state_bank import StateBank, available_views, compose_view


class StatePipeline:
    """Compatibility wrapper for state retrieval utilities."""

    def __init__(self, config, state_bank: StateBank, embedder, api_pool=None):
        self.config = config
        self.state_bank = state_bank
        self.embedder = embedder
        self.api_pool = api_pool

    async def retrieve_view(self, elements: Dict[str, Any], view: str) -> List[Dict[str, Any]]:
        """Retrieve one view and return ranked candidates."""
        composed = compose_view(elements, view)
        query_emb = await self.embedder.encode_view(composed)
        return self.state_bank.search_view(
            view=view,
            query_emb=query_emb,
            top_k=self.config.experience_top_n,
            min_score=self.config.min_score,
            min_q=self.config.min_q_value,
        )

    async def retrieve(self, elements: Dict[str, Any]) -> str:
        """Backward-compatible retrieval using a single broad view."""
        avail = available_views(elements)
        if not avail:
            return ""
        view = "all" if "all" in avail else avail[0]
        candidates = await self.retrieve_view(elements, view)
        selected = candidates[: int(getattr(self.config, "experience_top_n", 1))]
        return self.format_experiences(selected)

    @staticmethod
    def format_experiences(candidates: List[Dict[str, Any]]) -> str:
        """Format experiences into a prompt-friendly block."""
        entries = []
        for i, c in enumerate(candidates, 1):
            exp = c.get("experience", "")
            if not exp:
                continue
            source = c.get("source", "correct")
            tag = "Learned from success" if source == "correct" else "Learned from failure"
            entries.append(f"#{i} [{tag}]\n{exp}")
        return "\n\n".join(entries)

    async def close(self) -> None:
        """Release resources (no-op)."""
        return None
