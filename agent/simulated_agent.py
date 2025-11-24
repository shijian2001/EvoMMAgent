"""Example agent implementation with simulated LLM responses."""

from typing import List, Dict, Optional
from agent.base_agent import BasicAgent


class SimulatedAgent(BasicAgent):
    """A simulated agent that uses predefined responses for testing purposes.
    
    Instead of calling a real LLM, this agent uses predefined responses
    to demonstrate the ReAct pattern with tool usage.
    """
    
    def __init__(
            self,
            name: str = "SimulatedAgent",
            description_en: str = "A simulated agent for testing",
            tool_bank: Optional[List] = None,
            use_zh: bool = False,
            simulated_responses: Optional[List[str]] = None,
    ):
        """Initialize the simulated agent.
        
        Args:
            name: Agent name
            description_en: Agent description
            tool_bank: List of tools
            use_zh: Whether to use Chinese
            simulated_responses: List of simulated LLM responses for testing
        """
        super().__init__(
            name=name,
            description_en=description_en,
            tool_bank=tool_bank,
            use_zh=use_zh,
        )
        self.simulated_responses = simulated_responses or []
        self.response_idx = 0
    
    def _simulate_llm_response(self) -> str:
        """Get next simulated LLM response.
        
        Returns:
            Simulated response string
        """
        if self.response_idx < len(self.simulated_responses):
            response = self.simulated_responses[self.response_idx]
            self.response_idx += 1
            return response
        return "Task completed."
    
    async def act(
            self,
            task: str,
            max_iterations: int = 5,
            verbose: bool = True
    ) -> Dict:
        """Execute a task using ReAct pattern with simulated responses.
        
        Args:
            task: Task description
            max_iterations: Maximum number of iterations
            verbose: Whether to print execution steps
            
        Returns:
            Dict containing execution history and final result
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task}")
            print(f"{'='*60}\n")
        
        history = []
        
        for iteration in range(max_iterations):
            if verbose:
                print(f"--- Iteration {iteration + 1} ---")
            
            # Get simulated LLM response
            response = self._simulate_llm_response()
            
            if verbose:
                print(f"Agent: {response}")
            
            # Detect if response contains a tool call
            has_tool, tool_name, tool_args, thought = self._detect_tool(response)
            
            if has_tool:
                # Execute tool
                if verbose:
                    print(f"Calling tool: {tool_name}")
                    print(f"Arguments: {tool_args}")
                
                observation = self._call_tool(tool_name, tool_args)
                
                if verbose:
                    print(f"Observation: {observation}\n")
                
                history.append({
                    "iteration": iteration + 1,
                    "thought": thought,
                    "action": tool_name,
                    "action_input": tool_args,
                    "observation": observation,
                })
            else:
                # No tool call, agent finished
                history.append({
                    "iteration": iteration + 1,
                    "final_response": response,
                })
                
                if verbose:
                    print(f"\n{'='*60}")
                    print("Task completed!")
                    print(f"{'='*60}\n")
                
                break
        
        return {
            "task": task,
            "history": history,
            "success": True,
        }

