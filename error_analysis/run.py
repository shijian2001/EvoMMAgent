"""CLI entry point for error analysis and comparison."""

import argparse
import asyncio
import logging
import os

from .analyzer import CompareAnalyzer, ErrorAnalyzer
from .client import DEFAULT_API_URL, AsyncLLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trajectory error & comparison analysis via external LLM",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--api_key",
        default=os.environ.get("ANALYSIS_API_KEY", ""),
        help="API key (or set ANALYSIS_API_KEY env var)",
    )
    common.add_argument("--api_url", default=DEFAULT_API_URL, help="API endpoint URL")
    common.add_argument("--output_dir", required=True, help="Output directory")
    common.add_argument("--concurrency", type=int, default=10, help="Max concurrent LLM calls")

    # Error analysis sub-command
    err = sub.add_parser("error", parents=[common], help="Analyze incorrect w/tool cases")
    err.add_argument("--results", required=True, help="Path to results.jsonl")
    err.add_argument("--memory_dir", required=True, help="Path to memory directory")

    # Comparison sub-command
    cmp = sub.add_parser("compare", parents=[common], help="Compare direct vs w/tool")
    cmp.add_argument("--direct_results", required=True, help="Direct (no tools) results.jsonl")
    cmp.add_argument("--tool_results", required=True, help="W/tool results.jsonl")
    cmp.add_argument("--tool_memory_dir", required=True, help="W/tool memory directory (contains tasks/*/trace.json)")

    return parser


async def _run(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise ValueError("API key required: pass --api_key or set ANALYSIS_API_KEY")

    async with AsyncLLMClient(
        api_key=args.api_key,
        api_url=args.api_url,
        concurrency=args.concurrency,
    ) as client:
        if args.command == "error":
            analyzer = ErrorAnalyzer(client)
            await analyzer.analyze_all(args.results, args.memory_dir, args.output_dir)
        elif args.command == "compare":
            analyzer = CompareAnalyzer(client)
            await analyzer.analyze_all(
                args.direct_results, args.tool_results,
                args.tool_memory_dir, args.output_dir,
            )


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
