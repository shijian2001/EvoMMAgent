"""Error analysis and comparison analysis via external LLM."""

import asyncio
import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .client import AsyncLLMClient
from .formatter import (
    build_trace_index,
    format_trace,
    load_results,
    load_trace,
    resolve_image_paths,
)
from .prompts import COMPARE_ANALYSIS_PROMPT, ERROR_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)

ERROR_CATEGORIES = [
    "visual_perception",
    "ineffective_tool_use",
    "tool_misinterpretation",
    "reasoning_error",
    "instruction_following",
    "no_answer",
]


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #


def _parse_json_response(text: str) -> Dict:
    """Extract JSON object from LLM response (tolerant of markdown fences)."""
    for candidate in [
        text,
        # ```json ... ```
        (re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL) or type("", (), {"group": lambda s, i: ""})()).group(1),
        # first { ... }
        (re.search(r"\{.*\}", text, re.DOTALL) or type("", (), {"group": lambda s: ""})()).group(),
    ]:
        if candidate:
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
    return {}


# ------------------------------------------------------------------ #
#  Error Analyzer                                                     #
# ------------------------------------------------------------------ #


class ErrorAnalyzer:
    """Analyze incorrect w/tool cases by sending them to an external LLM."""

    def __init__(self, client: AsyncLLMClient):
        self.client = client

    async def analyze_all(
        self,
        results_path: str,
        memory_dir: str,
        output_dir: str,
    ) -> List[Dict]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        results = load_results(results_path)
        incorrect = [r for r in results if not r.get("is_correct", True)]
        logger.info("Total: %d, Incorrect: %d", len(results), len(incorrect))

        if not incorrect:
            logger.info("No incorrect cases to analyze.")
            return []

        trace_index = build_trace_index(memory_dir)

        total = len(incorrect)
        counter = {"done": 0}
        lock = asyncio.Lock()

        jsonl_path = output / "error_details.jsonl"
        jsonl_file = open(jsonl_path, "w", encoding="utf-8")
        state = {"done": 0, "file": jsonl_file, "records": []}
        lock = asyncio.Lock()

        logger.info("Starting error analysis for %d cases (concurrency=%d)...",
                     total, self.client._semaphore._value)
        await asyncio.gather(
            *(self._analyze_one(r, trace_index, state, lock, total) for r in incorrect)
        )
        jsonl_file.close()

        records = state["records"]
        self._write_report(output / "report.md", len(results), records)
        logger.info("Saved %d error analyses to %s", len(records), output)
        return records

    async def _analyze_one(
        self, result: Dict, trace_index: Dict,
        state: Dict, lock: asyncio.Lock, total: int,
    ) -> None:
        idx = result["idx"]
        trace_path = trace_index.get(str(idx))
        if not trace_path:
            logger.warning("[idx=%s] Trace not found, skipping", idx)
            return

        trace_data = load_trace(trace_path)
        formatted = format_trace(trace_data.get("trace", []))
        image_paths = resolve_image_paths(trace_data, trace_path.parent)
        image_labels = ", ".join(f"img_{i}" for i in range(len(image_paths)))

        prompt = ERROR_ANALYSIS_PROMPT.format(
            question=result.get("query", result.get("question", "")),
            choices=", ".join(result.get("choices", [])),
            ground_truth=result.get("ground_truth", ""),
            predicted=result.get("predicted", ""),
            image_labels=image_labels,
            formatted_trace=formatted,
        )

        try:
            response = await self.client.call(prompt, image_paths)
            parsed = _parse_json_response(response)
        except Exception as e:
            logger.error("[idx=%s] LLM call failed: %s", idx, e)
            parsed = {"error_categories": [], "analysis": f"FAILED: {e}"}

        record = {
            "idx": idx,
            "dataset": result.get("dataset", ""),
            "sub_task": result.get("sub_task", ""),
            "question": result.get("question", ""),
            "image_paths": [
                img.get("path", "")
                for img in trace_data.get("input", {}).get("images", [])
            ],
            "ground_truth": result.get("ground_truth", ""),
            "predicted": result.get("predicted", ""),
            "error_categories": parsed.get("error_categories", []),
            "analysis": parsed.get("analysis", ""),
        }

        async with lock:
            state["done"] += 1
            state["records"].append(record)
            state["file"].write(json.dumps(record, ensure_ascii=False) + "\n")
            state["file"].flush()
            logger.info("[%d/%d] idx=%s done", state["done"], total, idx)

    # ---- Report generation ---- #

    def _write_report(self, path: Path, total: int, records: List[Dict]) -> None:
        n = len(records)
        cat_counter: Counter = Counter()
        sub_task_data: Dict[str, Dict] = {}

        for r in records:
            cats = r.get("error_categories", [])
            for c in cats:
                cat_counter[c] += 1
            st = r.get("sub_task", "Unknown")
            sub_task_data.setdefault(st, {"count": 0, "cats": Counter()})
            sub_task_data[st]["count"] += 1
            for c in cats:
                sub_task_data[st]["cats"][c] += 1

        lines = [
            "# Error Analysis Report\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total samples | {total} |",
            f"| Incorrect | {n} ({n/total*100:.1f}%) |" if total else f"| Incorrect | {n} |",
            f"| Analyzed | {n} |",
            f"| Generated | {datetime.now().strftime('%Y-%m-%d %H:%M')} |",
            "",
            "## Error Category Distribution\n",
            "| Category | Count | % of Errors |",
            "|----------|-------|-------------|",
        ]
        for cat in ERROR_CATEGORIES:
            cnt = cat_counter.get(cat, 0)
            pct = cnt / n * 100 if n else 0
            lines.append(f"| {cat} | {cnt} | {pct:.1f}% |")
        lines += ["", "> Percentages may sum to >100% due to multi-label annotation.", ""]

        lines += [
            "## Breakdown by Sub-task\n",
            "| Sub-task | Errors | Top Categories |",
            "|----------|--------|----------------|",
        ]
        for st, d in sorted(sub_task_data.items()):
            top = ", ".join(f"{c} ({cnt})" for c, cnt in d["cats"].most_common(3))
            lines.append(f"| {st} | {d['count']} | {top} |")
        lines += [""]

        lines.append("## Case Details\n")
        for i, r in enumerate(records, 1):
            cats = ", ".join(r.get("error_categories", [])) or "N/A"
            imgs = ", ".join(r.get("image_paths", []))
            lines += [
                f"### #{i} [idx={r['idx']}] {r.get('sub_task', '')}",
                f"- **Question**: {r.get('question', '')}",
                f"- **Images**: {imgs}",
                f"- **Ground Truth**: {r.get('ground_truth', '')} | **Predicted**: {r.get('predicted', '')}",
                f"- **Categories**: {cats}",
                f"- **Analysis**: {r.get('analysis', '')}",
                "",
            ]

        path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ #
#  Compare Analyzer                                                   #
# ------------------------------------------------------------------ #


class CompareAnalyzer:
    """Analyze disagreements between direct and w/tool approaches."""

    def __init__(self, client: AsyncLLMClient):
        self.client = client

    async def analyze_all(
        self,
        direct_results_path: str,
        tool_results_path: str,
        memory_dir: str,
        output_dir: str,
    ) -> List[Dict]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        direct_dict = {r["idx"]: r for r in load_results(direct_results_path)}
        tool_dict = {r["idx"]: r for r in load_results(tool_results_path)}

        # Keep only one-correct-one-wrong disagreements
        disagreements = [
            (direct_dict[idx], tool_dict[idx])
            for idx in direct_dict
            if idx in tool_dict
            and direct_dict[idx].get("is_correct") != tool_dict[idx].get("is_correct")
        ]
        logger.info(
            "Total matched: %d, Disagreements: %d",
            len(set(direct_dict) & set(tool_dict)), len(disagreements),
        )

        if not disagreements:
            logger.info("No disagreements to analyze.")
            return []

        trace_index = build_trace_index(memory_dir)

        total = len(disagreements)
        counter = {"done": 0}
        lock = asyncio.Lock()

        jsonl_path = output / "compare_details.jsonl"
        jsonl_file = open(jsonl_path, "w", encoding="utf-8")
        state = {"done": 0, "file": jsonl_file, "records": []}
        lock = asyncio.Lock()

        logger.info("Starting comparison analysis for %d cases (concurrency=%d)...",
                     total, self.client._semaphore._value)
        await asyncio.gather(
            *(self._analyze_one(d, t, trace_index, state, lock, total)
              for d, t in disagreements)
        )
        jsonl_file.close()

        records = state["records"]
        self._write_report(
            output / "report.md",
            len(set(direct_dict) & set(tool_dict)),
            records,
        )
        logger.info("Saved %d comparison analyses to %s", len(records), output)
        return records

    async def _analyze_one(
        self, direct: Dict, tool: Dict, trace_index: Dict,
        state: Dict, lock: asyncio.Lock, total: int,
    ) -> None:
        idx = direct["idx"]
        trace_path = trace_index.get(str(idx))
        if not trace_path:
            logger.warning("[idx=%s] Trace not found, skipping", idx)
            return

        trace_data = load_trace(trace_path)
        formatted = format_trace(trace_data.get("trace", []))
        image_paths = resolve_image_paths(trace_data, trace_path.parent)
        image_labels = ", ".join(f"img_{i}" for i in range(len(image_paths)))

        prompt = COMPARE_ANALYSIS_PROMPT.format(
            question=direct.get("query", direct.get("question", "")),
            choices=", ".join(direct.get("choices", [])),
            ground_truth=direct.get("ground_truth", ""),
            image_labels=image_labels,
            direct_predicted=direct.get("predicted", ""),
            direct_response=direct.get("response", ""),
            tool_predicted=tool.get("predicted", ""),
            formatted_trace=formatted,
        )

        try:
            response = await self.client.call(prompt, image_paths)
            parsed = _parse_json_response(response)
        except Exception as e:
            logger.error("[idx=%s] LLM call failed: %s", idx, e)
            parsed = {
                "correct_approach": "unknown",
                "key_difference": f"FAILED: {e}",
                "explanation": "",
            }

        record = {
            "idx": idx,
            "dataset": direct.get("dataset", ""),
            "sub_task": direct.get("sub_task", ""),
            "question": direct.get("question", ""),
            "image_paths": [
                img.get("path", "")
                for img in trace_data.get("input", {}).get("images", [])
            ],
            "ground_truth": direct.get("ground_truth", ""),
            "direct_predicted": direct.get("predicted", ""),
            "direct_is_correct": direct.get("is_correct", False),
            "tool_predicted": tool.get("predicted", ""),
            "tool_is_correct": tool.get("is_correct", False),
            "correct_approach": parsed.get("correct_approach", ""),
            "key_difference": parsed.get("key_difference", ""),
            "explanation": parsed.get("explanation", ""),
        }

        async with lock:
            state["done"] += 1
            state["records"].append(record)
            state["file"].write(json.dumps(record, ensure_ascii=False) + "\n")
            state["file"].flush()
            logger.info("[%d/%d] idx=%s done", state["done"], total, idx)

    # ---- Report generation ---- #

    def _write_report(self, path: Path, total: int, records: List[Dict]) -> None:
        n = len(records)
        direct_wins = sum(1 for r in records if r.get("direct_is_correct"))
        tool_wins = sum(1 for r in records if r.get("tool_is_correct"))

        sub_task_data: Dict[str, Dict] = {}
        for r in records:
            st = r.get("sub_task", "Unknown")
            sub_task_data.setdefault(st, {"direct_wins": 0, "tool_wins": 0})
            if r.get("direct_is_correct"):
                sub_task_data[st]["direct_wins"] += 1
            else:
                sub_task_data[st]["tool_wins"] += 1

        lines = [
            "# Comparison Analysis Report\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total samples | {total} |",
            f"| Disagreements | {n} ({n/total*100:.1f}%) |" if total else f"| Disagreements | {n} |",
            f"| Direct wins | {direct_wins} |",
            f"| Tool wins | {tool_wins} |",
            f"| Generated | {datetime.now().strftime('%Y-%m-%d %H:%M')} |",
            "",
            "## Breakdown by Sub-task\n",
            "| Sub-task | Direct Wins | Tool Wins |",
            "|----------|-------------|-----------|",
        ]
        for st, d in sorted(sub_task_data.items()):
            lines.append(f"| {st} | {d['direct_wins']} | {d['tool_wins']} |")
        lines += [""]

        lines.append("## Case Details\n")
        for i, r in enumerate(records, 1):
            winner = "Direct" if r.get("direct_is_correct") else "Tool"
            imgs = ", ".join(r.get("image_paths", []))
            lines += [
                f"### #{i} [idx={r['idx']}] {r.get('sub_task', '')} — {winner} wins",
                f"- **Question**: {r.get('question', '')}",
                f"- **Images**: {imgs}",
                f"- **Ground Truth**: {r.get('ground_truth', '')}",
                f"- **Direct**: {r.get('direct_predicted', '')} ({'correct' if r.get('direct_is_correct') else 'wrong'})",
                f"- **Tool**: {r.get('tool_predicted', '')} ({'correct' if r.get('tool_is_correct') else 'wrong'})",
                f"- **Key Difference**: {r.get('key_difference', '')}",
                f"- **Explanation**: {r.get('explanation', '')}",
                "",
            ]

        path.write_text("\n".join(lines), encoding="utf-8")
