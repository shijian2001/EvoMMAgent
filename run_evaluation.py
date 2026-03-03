import asyncio
from runner import Runner


async def main():
    """Run evaluation on a dataset."""
    mode = "test" # train, test
    dataset = "hr_bench"
    dataset_jsonl = "/mnt/tidalfs-bdsz01/usr/wangshijian/EvoMMAgent/data/eval/image/benchmark/HR_Bench/hr_bench_data_meta_test.jsonl"
    dataset_image_dir = "/mnt/tidalfs-bdsz01/usr/wangshijian/EvoMMAgent/data/eval/image/benchmark/HR_Bench/images"
    # variant = "w_state_gpt4o" # direct, w_tool, w_trace_gpt4o, w_state_gpt4o
    ablation = 5
    
    # ============================================================
    # Example 1: Direct Query (no tools, no memory)
    # ============================================================
    # agent_config = {
    #     "tool_bank": None,  # No tools
    #     "model_name": "qwen3-vl-32b-instruct", # qwen2.5-vl-72b-instruct, qwen-2.5-vl-72b-instruct, qwen3-vl-235b-a22b-instruct, qwen3-vl-8b-instruct, qwen3-vl-32b-instruct
    #     "max_tokens": 40000,
    #     "temperature": 0.7,
    #     "enable_memory": False,  # No memory
    #     "max_iterations": 20,
    #     "max_retries": 20,
    #     "base_url": "https://maas.devops.xiaohongshu.com/v1",
    #     "api_keys": ["MAAS369f45faf38a4db59ae7dc6ed954a399", "MAASace45968cdbf4afeb71d07ecef846c94"],
    # }
    
    # ============================================================
    # Example 2: MMAgent with Tools (ReAct pattern)
    # ============================================================
    # agent_config = {
    #     "tool_bank": [
    #         "ocr",
    #         "solve_math_equation",
    #         "web_search",
    #         "localize_objects", 
    #         "zoom_in", 
    #         "calculator", 
    #         "crop",
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
    #     "memory_dir": f"/mnt/tidalfs-bdsz01/dataset/llm_dataset/shijian/evommagent/memory/exp/meta_{mode}/{dataset}/qwen3vl_32b/w_tool",
    #     "max_retries": 20,
    #     "mm_agent_template_en_file": "exp_prompt/base/en.jinja2",
    #     "mm_agent_template_zh_file": "exp_prompt/base/zh.jinja2",
    #     "base_url": "https://maas.devops.xiaohongshu.com/v1",
    #     "api_keys": ["MAAS369f45faf38a4db59ae7dc6ed954a399", "MAASace45968cdbf4afeb71d07ecef846c94"],
    #     # "base_url": "http://10.217.65.160:8000/v1",
    #     # "api_keys": ["dummy key"],
    # }
    
    # ============================================================
    # Example 3: MMAgent with Tools + Trace-Level Retrieval
    # ============================================================
    # agent_config = {
    #     "tool_bank": [
    #         "ocr",
    #         "solve_math_equation",
    #         "web_search",
    #         "localize_objects", 
    #         "zoom_in", 
    #         "calculator", 
    #         "crop",
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
    #     "memory_dir": f"/mnt/tidalfs-bdsz01/dataset/llm_dataset/shijian/evommagent/memory/exp/meta_test/{dataset}/qwen3vl_32b/w_trace_gpt4o",
    #     "max_iterations": 20,
    #     "max_retries": 20,
    #     "mm_agent_template_en_file": "exp_prompt/base/en.jinja2",
    #     "mm_agent_template_zh_file": "exp_prompt/base/zh.jinja2",
    #     "base_url": "https://maas.devops.xiaohongshu.com/v1",
    #     "api_keys": ["MAAS369f45faf38a4db59ae7dc6ed954a399", "MAASace45968cdbf4afeb71d07ecef846c94"],
    #     # Trace-level retrieval: one-shot before ReAct loop
    #     "retrieval": {
    #         "enable": True,
    #         "mode": "trace",
    #         "bank_memory_dir": f"/mnt/tidalfs-bdsz01/dataset/llm_dataset/shijian/evommagent/memory/exp/meta_train/{dataset}/qwen3vl_32b/w_tool",
    #         "bank_dir_name": "trace_bank_gpt4o",
    #         "embedding_model": "qwen3vl-embed",
    #         "embedding_base_url": "http://localhost:8001/v1",
    #         "trace_top_n": 1,
    #         "min_score": 0.1,
    #     },
    # }
    
    # ============================================================
    # Example 4: MMAgent with Tools + State-Level Retrieval (MDP)
    # ============================================================
    agent_config = {
        "tool_bank": [
            "ocr",
            "solve_math_equation",
            "web_search",
            "localize_objects", 
            "zoom_in", 
            "calculator", 
            "crop",
            "visualize_regions",
            "estimate_region_depth", 
            "estimate_object_depth", 
            "get_image2images_similarity", 
            "get_image2texts_similarity", 
            "get_text2images_similarity"
        ],
        "model_name": "qwen3-vl-32b-instruct",
        "max_tokens": 40000,
        "temperature": 0.7,
        "enable_memory": False,
        "memory_dir": f"/mnt/tidalfs-bdsz01/dataset/llm_dataset/shijian/evommagent/memory/exp/meta_test/{dataset}/qwen3vl_32b/ablation/ablation_wide/max_epoch=3/wide_{ablation}", 
        # {variant}
        # ablation/ablation_deep/experience_top_n=3/deep_{ablation}
        # ablation/ablation_wide/max_epoch=3/wide_{ablation}
        "max_iterations": 20,
        "max_retries": 20,
        "mm_agent_template_en_file": "exp_prompt/base/en.jinja2",
        "mm_agent_template_zh_file": "exp_prompt/base/zh.jinja2",
        "base_url": "https://maas.devops.xiaohongshu.com/v1",
        "api_keys": ["MAAS369f45faf38a4db59ae7dc6ed954a399", "MAASace45968cdbf4afeb71d07ecef846c94"],
        # State-level retrieval: per-step experience from hindsight-annotated states
        "retrieval": {
            "enable": True,
            "mode": "state",
            "bank_memory_dir": f"/mnt/tidalfs-bdsz01/dataset/llm_dataset/shijian/evommagent/memory/exp/meta_train/{dataset}/qwen3vl_32b/w_tool",
            "bank_dir_name": "state_bank_gpt4o",
            "embedding_model": "qwen3vl-embed",
            "embedding_base_url": "http://localhost:8001/v1",
            "min_score": 0.1,
            "max_epoch": 3,
            "min_q_value": 5,
            "experience_top_n": ablation,
        },
    }
    
    # Runner configuration
    runner = Runner(
        jsonl_path=dataset_jsonl,
        image_dir=dataset_image_dir,
        agent_config=agent_config,
        output_dir=f"eval_results/exp/meta_{mode}/{dataset}/qwen3vl_32b/ablation/ablation_wide/max_epoch=3/wide_{ablation}",
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
