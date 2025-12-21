"""Example script for running evaluation."""

import asyncio
from runner import Runner


async def main():
    """Run evaluation on a dataset."""
    
    # ============================================================
    # Example 1: Direct Query (no tools, no memory)
    # ============================================================
    # agent_config = {
    #     "tool_bank": None,  # No tools
    #     "model_name": "qwen2.5-vl-72b-instruct",
    #     "max_tokens": 2048,
    #     "temperature": 0.0,  # Use 0 for deterministic evaluation
    #     "enable_memory": False,  # No memory
    #     "use_zh": False,
    # }
    
    # ============================================================
    # Example 2: MMAgent with Tools (ReAct pattern)
    # ============================================================
    agent_config = {
        "tool_bank": ["crop", "get_objects", "ocr"],  # Add tools as needed
        "model_name": "qwen2.5-vl-72b-instruct",
        "max_tokens": 2048,
        "temperature": 0.0,  # Use 0 for deterministic evaluation
        "enable_memory": True,
        "memory_dir": "memory",
        "use_zh": False,
        "mm_agent_template_en_file": "Eval_MMAgent_EN.jinja2",  # Use MMAgent template
        "mm_agent_template_zh_file": "Eval_MMAgent_ZH.jinja2",
    }
    
    # Runner configuration
    runner = Runner(
        jsonl_path="path/to/data.jsonl",
        image_dir="path/to/images",
        agent_config=agent_config,
        output_dir="eval_results/dataset_name",
        batch_size=32,
        max_concurrent=16,
        verbose=True
    )
    
    # Run evaluation
    stats = await runner.run_evaluation()
    
    # Print statistics
    print("\n" + "="*80)
    print("📊 Evaluation Statistics")
    print("="*80)
    print(f"Overall Accuracy: {stats['accuracy']*100:.2f}%")
    print(f"Total: {stats['total']}, Correct: {stats['correct']}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
