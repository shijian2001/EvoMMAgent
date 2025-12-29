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
            "en": f"You may think step-by-step to analyze the question, but you MUST end your response with your final answer in this EXACT format on a new line: {special_answer_token} [single letter only, e.g., A or B or C]. Do not add any explanation after the answer.",
            "zh": f"你可以逐步思考来分析问题，但必须在回答的最后一行按以下精确格式输出最终答案：{special_answer_token} [仅单个字母，如 A 或 B 或 C]。答案后不要添加任何解释。",
        },
        "open-ended": {
            "en": f"You may provide reasoning first, but you MUST end your response with your final answer in this format on a new line: {special_answer_token} [your concise answer]",
            "zh": f"你可以先提供推理过程，但必须在回答的最后一行按以下格式输出最终答案：{special_answer_token} [你的简洁答案]",
        },
        "yes-no": {
            "en": f"You may explain your reasoning, but you MUST end with: {special_answer_token} Yes or {special_answer_token} No",
            "zh": f"你可以解释推理过程，但必须以以下格式结尾：{special_answer_token} Yes 或 {special_answer_token} No",
        },
        "true-false": {
            "en": f"You may explain your reasoning, but you MUST end with: {special_answer_token} True or {special_answer_token} False",
            "zh": f"你可以解释推理过程，但必须以以下格式结尾：{special_answer_token} True 或 {special_answer_token} False",
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
