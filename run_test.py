"""Test script to demonstrate EvoMMAgent functionality."""

import asyncio

from tool.calculator_tool import CalculatorTool
from agent.simulated_agent import SimulatedAgent


async def test_agent_with_tool():
    """Test the agent using calculator tool."""
    print("\n" + "="*60)
    print("EvoMMAgent Test: Calculator Agent")
    print("="*60 + "\n")
    
    # Create agent with calculator tool
    agent = SimulatedAgent(
        name="MathAgent",
        description_en="An agent that can perform calculations",
        tool_bank=["calculator"],
        simulated_responses=[
            "I need to calculate 15 + 27.\nAction: calculator\nAction Input: {\"operation\": \"add\", \"a\": 15, \"b\": 27}\nObservation:",
            "Now multiply by 2.\nAction: calculator\nAction Input: {\"operation\": \"multiply\", \"a\": 42, \"b\": 2}\nObservation:",
            "The answer is 84.",
        ]
    )
    
    # Execute task
    result = await agent.act(
        task="Calculate (15 + 27) * 2",
        max_iterations=5,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("✓ Test completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(test_agent_with_tool())
