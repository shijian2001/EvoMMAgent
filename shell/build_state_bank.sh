#!/bin/bash

# 使用方式: ./build_state_bank.sh <llm_model>
# 示例: ./build_state_bank.sh qwen3-vl-235b-a22b-instruct
# 示例: ./build_state_bank.sh qwen3-vl-32b-instruct

set -e  # 遇到错误时退出

# 获取外部传入的llm_model参数
LLM_MODEL=${1:-"qwen3-vl-235b-a22b-instruct"}

# 根据llm_model设置bank_dir_name
case $LLM_MODEL in
    "qwen3-vl-235b-a22b-instruct")
        BANK_DIR_NAME="state_bank_qwen3vl235b"
        ;;
    "qwen3-vl-32b-instruct")
        BANK_DIR_NAME="state_bank_qwen3vl32b"
        ;;
    *)
        echo "❌ Unsupported llm_model: $LLM_MODEL"
        echo "Supported models:"
        echo "  - qwen3-vl-235b-a22b-instruct"
        echo "  - qwen3-vl-32b-instruct"
        exit 1
        ;;
esac

echo "=========================================="
echo "LLM Model: $LLM_MODEL"
echo "Bank Dir Name: $BANK_DIR_NAME"
echo "=========================================="

# memory_dir列表 - 在这里添加/修改路径
MEMORY_DIRS=(
    "/mnt/tidalfs-bdsz01/dataset/llm_dataset/shijian/evommagent/memory/exp/meta_train/mathverse_mc/qwen3vl_32b/w_tool"
    "/mnt/tidalfs-bdsz01/dataset/llm_dataset/shijian/evommagent/memory/exp/meta_train/mmmu_mc/qwen3vl_32b/w_tool"
    "/mnt/tidalfs-bdsz01/dataset/llm_dataset/shijian/evommagent/memory/exp/meta_train/mmstar/qwen3vl_32b/w_tool"
    "/mnt/tidalfs-bdsz01/dataset/llm_dataset/shijian/evommagent/memory/exp/meta_train/wemath/qwen3vl_32b/w_tool"
)

TOTAL=${#MEMORY_DIRS[@]}
CURRENT=0

# 串行执行
for MEMORY_DIR in "${MEMORY_DIRS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "=========================================="
    echo "[$CURRENT/$TOTAL] Processing: $MEMORY_DIR"
    echo "=========================================="
    
    python scripts/build_state_bank.py \
        --memory_dir "$MEMORY_DIR" \
        --llm_model "$LLM_MODEL" \
        --llm_base_url https://maas.devops.xiaohongshu.com/v1 \
        --llm_api_key MAAS369f45faf38a4db59ae7dc6ed954a399 \
        --embedding_model qwen3vl-embed \
        --embedding_base_url http://localhost:8001/v1 \
        --concurrency 16 \
        --mode both \
        --batch_size 64 \
        --bank_dir_name "$BANK_DIR_NAME"
    
    echo "✅ [$CURRENT/$TOTAL] Completed: $MEMORY_DIR"
done

echo ""
echo "=========================================="
echo "✅ All $TOTAL tasks completed successfully!"
echo "=========================================="
