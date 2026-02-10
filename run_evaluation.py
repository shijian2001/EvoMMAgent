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
        "model_name": "qwen3-vl-32b-instruct", # qwen2.5-vl-72b-instruct, qwen-2.5-vl-72b-instruct, qwen3-vl-235b-a22b-instruct, qwen3-vl-8b-instruct, qwen3-vl-32b-instruct
        "max_tokens": 40000,
        "temperature": 0.7,
        "enable_memory": False,  # No memory
        "max_iterations": 20,
        "max_retries": 20,
        "base_url": "https://maas.devops.xiaohongshu.com/v1",
        "api_keys": ["MAAS369f45faf38a4db59ae7dc6ed954a399"],
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
    #     "model_name": "qwen3-vl-32b-instruct",
    #     "max_tokens": 40000,
    #     "temperature": 0.7,
    #     "enable_memory": False,
    #     "max_iterations": 20,
    #     "memory_dir": "/mnt/tidalfs-bdsz01/dataset/llm_dataset/shijian/evommagent/memory/20260209/blink/qwen3vl_32b/w_tool",
    #     "max_retries": 20,
    #     "mm_agent_template_en_file": "exp_prompt/more_tool_call/en.jinja2",
    #     "mm_agent_template_zh_file": "exp_prompt/more_tool_call/zh.jinja2",
    #     "base_url": "https://maas.devops.xiaohongshu.com/v1",
    #     "api_keys": ["MAAS369f45faf38a4db59ae7dc6ed954a399"],
    #     # "base_url": "http://10.217.65.160:8000/v1",
    #     # "api_keys": ["dummy key"],
    # }
    
    # ============================================================
    # Example 3: MMAgent with Tools + Retrieval (Self-Evolving)
    # ============================================================
    # agent_config = {
    #     "tool_bank": [
    #         "localize_objects", "zoom_in", "calculator", "crop",
    #         "visualize_regions", "estimate_region_depth",
    #         "estimate_object_depth", "get_image2images_similarity",
    #         "get_image2texts_similarity", "get_text2images_similarity",
    #     ],
    #     "model_name": "qwen3-vl-8b-instruct",
    #     "max_tokens": 40000,
    #     "temperature": 0.7,
    #     "enable_memory": False,
    #     "max_iterations": 20,
    #     "max_retries": 20,
    #     "mm_agent_template_en_file": "exp_prompt/more_tool_call/en.jinja2",
    #     "mm_agent_template_zh_file": "exp_prompt/more_tool_call/zh.jinja2",
    #     "base_url": "https://maas.devops.xiaohongshu.com/v1",
    #     "api_keys": [""],
    #     # Retrieval config — enable to use experience from training traces
    #     "retrieval": {
    #         "enable": True,
    #         "bank_memory_dir": "/path/to/training/memory",  # dir with tasks/ and bank/
    #         "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
    #         "embedding_base_url": "http://localhost:8001/v1",
    #         "rerank_model": "Qwen/Qwen3-VL-Reranker-2B",
    #         "rerank_base_url": "http://localhost:8002/v1",
    #         "enable_query_rewrite": True,
    #         "max_sub_queries": 3,
    #         "retrieval_top_k": 10,
    #         "enable_rerank": True,
    #         "rerank_top_n": 3,
    #         "min_score": 0.1,
    #         "max_retrieval_rounds": 1,
    #     },
    # }
    
    # Runner configuration
    runner = Runner(
        jsonl_path="data/eval/image/BLINK/blink_data.jsonl",
        image_dir="data/eval/image/BLINK/blink_images",
        agent_config=agent_config,
        output_dir="eval_results/20260209/blink/qwen3vl_32b/direct",
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
