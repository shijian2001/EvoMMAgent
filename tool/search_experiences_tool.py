"""Tool for state-level experience retrieval with multi-view search."""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from tool.base_tool import BasicTool, register_tool

from mm_memory.state_bank import ALL_VIEWS, VIEW_DESCRIPTIONS, available_views, compose_view


@register_tool(name="search_experiences")
class SearchExperiencesTool(BasicTool):
    """Search experiences from similar reasoning states."""

    name = "search_experiences"
    description_en = (
        "Search for experiences from similar reasoning states with multi-view retrieval. "
        "Each view matches by a different combination of state elements at varying granularities: "
        + ", ".join(f"{v} ({VIEW_DESCRIPTIONS[v]})" for v in ALL_VIEWS)
        + ". Call with different views to gather diverse perspectives, "
        "unless you are confident enough to make the best decision."
    )
    description_zh = (
        "通过多视角检索相似推理状态的经验。"
        "每个视角以不同粒度的状态元素组合进行匹配："
        + "、".join(f"{v}（{VIEW_DESCRIPTIONS[v]}）" for v in ALL_VIEWS)
        + "。使用不同视角调用以获取多元参考，除非你已有足够信心做出最佳决策。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "view": {
                "type": "string",
                "description": "The retrieval view to use: " + ", ".join(ALL_VIEWS),
                "enum": ALL_VIEWS,
            }
        },
        "required": ["view"],
    }
    example = '{"view":"question+images"}'

    def __init__(
        self,
        cfg: Optional[Dict] = None,
        use_zh: bool = False,
        *,
        state_bank,
        embedder,
        retrieval_config,
    ):
        super().__init__(cfg=cfg, use_zh=use_zh)
        self.state_bank = state_bank
        self.embedder = embedder
        self.retrieval_config = retrieval_config

        self._elements: Optional[Dict[str, Any]] = None
        self._available_views: List[str] = []
        self._used_views: Set[str] = set()
        self._round: int = 0

    def reset_state(self, elements: Dict[str, Any]) -> None:
        """Reset round/view tracking when entering a new state."""
        self._elements = elements
        self._available_views = available_views(elements)
        self._used_views = set()
        self._round = 0

    def call(self, params: Union[str, Dict]) -> Dict[str, Any]:
        raise NotImplementedError("search_experiences is async-only; use call_async.")

    async def call_async(self, params: Union[str, Dict]) -> Tuple[str, Dict[str, Any]]:
        """Run one retrieval round and return (observation_text, log_entry)."""
        params_dict = self.parse_params(params)
        view = str(params_dict.get("view", "")).strip()
        max_epoch = int(getattr(self.retrieval_config, "max_epoch", 1))
        top_n = int(getattr(self.retrieval_config, "experience_top_n", 1))

        if not self._elements:
            obs = "State context is not initialized. Proceed with your action or answer."
            return obs, {"round": self._round, "view": view, "experiences": []}

        if self._round >= max_epoch:
            obs = "Retrieval limit reached for this state. Proceed with your action or answer."
            return obs, {"round": self._round, "view": view, "experiences": []}

        remaining = [v for v in self._available_views if v not in self._used_views]
        if view not in remaining:
            remaining_text = ", ".join(remaining) if remaining else "(none)"
            obs = (
                f"View '{view}' is unavailable for this round. "
                f"Not-yet-used views: {remaining_text}. "
                "Proceed with your action or answer if guidance is sufficient."
            )
            return obs, {"round": self._round, "view": view, "experiences": []}

        composed = compose_view(self._elements, view)
        query_emb = await self.embedder.encode_multimodal(
            str(composed.get("text", "")),
            composed.get("images"),
        )
        candidates = self.state_bank.search_view(
            view=view,
            query_emb=query_emb,
            top_k=top_n,
            min_score=self.retrieval_config.min_score,
            min_q=self.retrieval_config.min_q_value,
        )

        self._used_views.add(view)
        self._round += 1
        remaining_after = [v for v in self._available_views if v not in self._used_views]
        is_final = self._round >= max_epoch or not remaining_after

        log_entry = {
            "round": self._round,
            "view": view,
            "experiences": [
                {
                    "experience": c.get("experience", ""),
                    "source": c.get("source", "correct"),
                    "score": c.get("retrieval_score", 0.0),
                    "task_id": c.get("task_id", ""),
                    "state": c.get("state", -1),
                }
                for c in candidates
                if c.get("experience", "")
            ],
        }
        observation = self._format_observation(
            round_index=self._round,
            max_epoch=max_epoch,
            candidates=candidates,
            remaining=remaining_after,
            is_final=is_final,
        )
        return observation, log_entry

    def _format_observation(
        self,
        round_index: int,
        max_epoch: int,
        candidates: List[Dict[str, Any]],
        remaining: List[str],
        is_final: bool,
    ) -> str:
        header = (
            "[Experiences from similar reasoning states]\n"
            "Ranked by relevance. Critically evaluate whether these experiences "
            "are helpful for making a good decision."
        )

        entries: List[str] = []
        for i, c in enumerate(candidates, 1):
            exp = c.get("experience", "")
            if not exp:
                continue
            source = c.get("source", "correct")
            tag = "Learned from success" if source == "correct" else "Learned from failure"
            entries.append(f"#{i} [{tag}]\n{exp}")

        if not entries:
            body = "No relevant experiences were found for this view."
        else:
            body = "\n\n".join(entries)

        if is_final:
            footer = "Retrieval complete. Proceed with your action or answer."
        else:
            remaining_block = "\n".join(f"- {v}" for v in remaining) if remaining else "(none)"
            footer = (
                "Consider searching with a different view for additional insights "
                "before acting, unless you are confident enough to make the best decision. "
                "Not-yet-used views:\n"
                f"{remaining_block}"
            )

        return f"{header}\n\n{body}\n\n{footer}"
