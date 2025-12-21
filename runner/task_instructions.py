"""Task-specific instructions for different question types."""

TASK_INSTRUCTIONS = {
    "multi-choice": {
        "en": "Select one option and output ONLY the option letter (A/B/C/D/E) after Answer:",
        "zh": "选择一个选项，Answer: 后仅输出选项字母（A/B/C/D/E）",
    },
    "open-ended": {
        "en": "Provide a concise answer in the format: Answer: [your answer]",
        "zh": "提供简洁答案，格式为：Answer: [你的答案]",
    },
    "yes-no": {
        "en": "Output only: Answer: Yes or Answer: No",
        "zh": "仅输出：Answer: Yes 或 Answer: No",
    },
    "true-false": {
        "en": "Output only: Answer: True or Answer: False",
        "zh": "仅输出：Answer: True 或 Answer: False",
    },
}


def get_task_instruction(task_type: str, use_zh: bool = False) -> str:
    """Get instruction text for a specific task type.
    
    Args:
        task_type: Type of task (multi-choice, open-ended, etc.)
        use_zh: Whether to use Chinese instruction
    
    Returns:
        Instruction text, or empty string if task type not found
    """
    lang = "zh" if use_zh else "en"
    return TASK_INSTRUCTIONS.get(task_type, {}).get(lang, "")
