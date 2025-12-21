"""Task-specific instructions for different question types."""


def get_task_instructions(special_answer_token: str = "\nAnswer:") -> dict:
    """Get all task instructions with the specified answer token.
    
    Args:
        special_answer_token: Token to mark final answer (default: "\nAnswer:")
    
    Returns:
        Dictionary of task instructions
    """
    return {
        "multi-choice": {
            "en": f"Select one option and output ONLY the option letter (A/B/C/D/E) in the format: {special_answer_token} [letter]",
            "zh": f"选择一个选项，仅输出选项字母（A/B/C/D/E），格式为：{special_answer_token} [字母]",
        },
        "open-ended": {
            "en": f"Provide a concise answer in the format: {special_answer_token} [your answer]",
            "zh": f"提供简洁答案，格式为：{special_answer_token} [你的答案]",
        },
        "yes-no": {
            "en": f"Output in the format: {special_answer_token} Yes or {special_answer_token} No",
            "zh": f"按格式输出：{special_answer_token} Yes 或 {special_answer_token} No",
        },
        "true-false": {
            "en": f"Output in the format: {special_answer_token} True or {special_answer_token} False",
            "zh": f"按格式输出：{special_answer_token} True 或 {special_answer_token} False",
        },
    }


def get_task_instruction(task_type: str, use_zh: bool = False, special_answer_token: str = "\nAnswer:") -> str:
    """Get instruction text for a specific task type.
    
    Args:
        task_type: Type of task (multi-choice, open-ended, etc.)
        use_zh: Whether to use Chinese instruction
        special_answer_token: Token to mark final answer (default: "\nAnswer:")
    
    Returns:
        Instruction text, or empty string if task type not found
    """
    lang = "zh" if use_zh else "en"
    instructions = get_task_instructions(special_answer_token)
    return instructions.get(task_type, {}).get(lang, "")
