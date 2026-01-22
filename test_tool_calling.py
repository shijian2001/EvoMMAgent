"""
完整 Tool Calling 流程验证脚本
模拟：用户消息 → 工具调用 → 工具结果 → 最终回复
"""
 
import asyncio
from openai import AsyncOpenAI
 
# ============ 配置 ============
BASE_URL = "https://maas.devops.xiaohongshu.com/v1"
API_KEY = "MAAS369f45faf38a4db59ae7dc6ed954a399"
# MAAS369f45faf38a4db59ae7dc6ed954a399
# QSTcac25b156c126d777c3c239a51cf941c
MODEL_NAME = "qwen3-vl-8b-instruct"
 
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Calculate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    }
}
 
async def test_full_tool_calling():
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    print("=" * 60)
    print("完整 Tool Calling 流程验证")
    print("=" * 60)
    
    # Step 1: 用户提问，模型调用工具
    print("\n📤 Step 1: 用户提问")
    response1 = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 123 * 456? Use the calculator."}
        ],
        tools=[CALCULATOR_TOOL],
        tool_choice="auto",
        temperature=0.0,
    )
    
    msg1 = response1.choices[0].message
    print(f"content: {msg1.content}")
    print(f"tool_calls: {msg1.tool_calls}")
    
    if not msg1.tool_calls:
        print("❌ 没有 tool_calls，无法继续测试")
        return
    
    tool_call = msg1.tool_calls[0]
    tool_call_id = tool_call.id
    
    # Step 2: 模拟工具执行结果，继续对话
    print("\n📤 Step 2: 发送工具结果，获取最终回复")
    response2 = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 123 * 456? Use the calculator."},
            {
                "role": "assistant",
                "content": msg1.content or "",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": "Result: 56088"
            }
        ],
        tools=[CALCULATOR_TOOL],
        tool_choice="auto",
        temperature=0.0,
    )
    
    msg2 = response2.choices[0].message
    print(f"content (display): {msg2.content}")
    print(f"content (repr):    {repr(msg2.content)}")
    
    # 检查特殊标记
    print("\n📊 诊断结果:")
    special_tokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    found = [t for t in special_tokens if t in (msg2.content or "")]
    
    if found:
        print(f"❌ 发现特殊标记: {found}")
        print("💡 这就是问题所在！vLLM 在 tool 消息后的回复中泄漏了聊天模板标记")
    else:
        print("✅ 未发现特殊标记")
    
    print("=" * 60)
 
if __name__ == "__main__":
    asyncio.run(test_full_tool_calling())