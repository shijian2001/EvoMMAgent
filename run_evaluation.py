"""Example script for running evaluation."""

import asyncio
from runner import Runner


async def main():
    """Run evaluation on a dataset."""
    
    # ============================================================
    # Example 1: Direct Query (no tools, no memory)
    # ============================================================
    agent_config = {
        "tool_bank": None,  # No tools
        "model_name": "qwen3-vl-8b-instruct", # qwen2.5-vl-72b-instruct, qwen-2.5-vl-72b-instruct, qwen3-vl-235b-a22b-instruct, qwen3-vl-8b-instruct, qwen3-vl-32b-instruct
        "max_tokens": 40000,
        "temperature": 0.7,
        "enable_memory": False,  # No memory
        "max_iterations": 20,
        "max_retries": 20,
        "base_url": "https://maas.devops.xiaohongshu.com/v1",
        "api_keys": ["MAAS369f45faf38a4db59ae7dc6ed954a399"],
        # "base_url": "https://maas.devops.xiaohongshu.com/v1",
        # "api_keys": ["MAAS369f45faf38a4db59ae7dc6ed954a399"],
    }
    
    # ============================================================
    # Example 2: MMAgent with Tools (ReAct pattern)
    # ============================================================
    # agent_config = {
    #     "tool_bank": [
    #         # "ocr",
    #         # "get_images",
    #         "localize_objects", 
    #         "zoom_in", 
    #         "calculator", 
    #         "crop",
    #         # "measure_region_property",
    #         "visualize_regions",
    #         "estimate_region_depth", 
    #         "estimate_object_depth", 
    #         "get_image2images_similarity", 
    #         "get_image2texts_similarity", 
    #         "get_text2images_similarity"
    #     ],
    #     "model_name": "qwen3-vl-8b-instruct",
    #     "max_tokens": 40000,
    #     "temperature": 0.7,
    #     "enable_memory": False,
    #     "max_iterations": 20,
    #     "memory_dir": "/mnt/tidalfs-bdsz01/dataset/llm_dataset/shijian/evommagent/memory/20260204/blink/qwen3vl_8b/w_tool",
    #     "max_retries": 20,
    #     "mm_agent_template_en_file": "exp_prompt/more_tool_call/en.jinja2",
    #     "mm_agent_template_zh_file": "exp_prompt/more_tool_call/zh.jinja2",
    #     "base_url": "https://maas.devops.xiaohongshu.com/v1",
    #     "api_keys": ["MAAS369f45faf38a4db59ae7dc6ed954a399"],
    #     # "base_url": "http://10.217.65.160:8000/v1",
    #     # "api_keys": ["dummy key"],
    # }
    
    # Runner configuration
    runner = Runner(
        jsonl_path="data/eval/image/BLINK/blink_data.jsonl",
        image_dir="data/eval/image/BLINK/blink_images",
        agent_config=agent_config,
        output_dir="eval_results/20260204/blink/qwen3vl_8b/direct",
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
