"""Prompt templates for error analysis and comparison."""

ERROR_ANALYSIS_PROMPT = """\
You are an expert evaluator analyzing why a multimodal AI agent answered a question incorrectly.

## Task Information
- Question: {question}
- Choices: {choices}
- Ground Truth: {ground_truth}
- Agent's Answer: {predicted}

The input images are provided in order: {image_labels}.

## Agent's Reasoning Trace
{formatted_trace}

## Error Categories (multi-label, select ALL that apply)
1. visual_perception — Agent or tools misidentified visual content (colors, objects, styles, spatial relations, etc.)
2. ineffective_tool_use — Wrong tool selected, redundant/repeated calls, wrong target, or failed to use a needed tool
3. tool_misinterpretation — Tool output was reasonable but agent drew incorrect conclusions from it
4. reasoning_error — Logical flaws in the reasoning chain despite correct observations
5. instruction_following — Did not follow required answer format, selected outside given options, or missed constraints
6. no_answer — Failed to produce a clear final answer

## Instructions
1. Examine the input images yourself to understand what the correct answer should be.
2. Review the agent's trace step by step, identifying where things went wrong.
3. Select ALL applicable error categories.
4. Write a concise, insightful root-cause analysis that can inform future improvements. Avoid redundancy.

Respond in this exact JSON format (no extra text):
{{"error_categories": ["category1", "category2"], "analysis": "..."}}"""


COMPARE_ANALYSIS_PROMPT = """\
You are an expert evaluator comparing two approaches to the same multimodal question.

## Task Information
- Question: {question}
- Choices: {choices}
- Ground Truth: {ground_truth}

The input images are provided in order: {image_labels}.

## Approach A: Direct (no tools)
- Answer: {direct_predicted}
- Response:
{direct_response}

## Approach B: With Tools
- Answer: {tool_predicted}
- Reasoning Trace:
{formatted_trace}

## Instructions
1. Examine the input images to understand the correct answer.
2. Analyze why the two approaches reached different conclusions.
3. Identify which approach was correct and explain the root cause of the other's failure.

Respond in this exact JSON format (no extra text):
{{"correct_approach": "direct" or "tool", "key_difference": "One sentence summarizing the core divergence.", "explanation": "Concisely explain why the approaches diverged and what caused the failure, with insights that can inform future improvements. Avoid redundancy."}}"""
