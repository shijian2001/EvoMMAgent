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
    # }
    
    # ============================================================
    # Example 2: MMAgent with Tools (ReAct pattern)
    # ============================================================
    agent_config = {
        "tool_bank": [
            "ocr",
            "get_images",
            "zoom_in", 
            "calculator", 
            "crop", 
            "visualize_regions",
            "localize_objects", 
            "estimate_region_depth", 
            "estimate_object_depth", 
            "get_image2images_similarity", 
            "get_image2texts_similarity", 
            "get_text2images_similarity"
        ],
        "model_name": "qwen-2.5-vl-72b-instruct",
        "max_tokens": 2048,
        "temperature": 0.0,  # Use 0 for deterministic evaluation
        "enable_memory": True,
        "memory_dir": "/mnt/tidalfs-bdsz01/dataset/llm_dataset/shijian/evommagent/memory/20251223",
        "max_retries": 10,
        "mm_agent_template_en_file": "Eval_MMAgent_EN.jinja2",  # Use MMAgent template
        "mm_agent_template_zh_file": "Eval_MMAgent_ZH.jinja2",
        "base_url": "http://localhost:8000/v1",
        "api_keys": ["dummy-key"],
    }
    
    # Runner configuration
    runner = Runner(
        jsonl_path="data/eval/image/BLINK/blink_data.jsonl",
        image_dir="data/eval/image/BLINK/blink_images",
        agent_config=agent_config,
        output_dir="eval_results/20251223/blink/qwen2.5_72b/ours",
        batch_size=100,
        max_concurrent=10,
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
