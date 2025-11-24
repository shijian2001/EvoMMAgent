"""Tool for solving mathematical equations using WolframAlpha."""

import os
from typing import Union, Dict
from tool.base_tool import BasicTool, register_tool

try:
    import wolframalpha
    WOLFRAM_AVAILABLE = True
except ImportError:
    WOLFRAM_AVAILABLE = False


@register_tool(name="solve_math_equation")
class SolveMathEquationTool(BasicTool):
    """A tool to solve mathematical equations using WolframAlpha."""
    
    name = "solve_math_equation"
    description_en = "Solve mathematical equations and problems using WolframAlpha. Can handle algebra, calculus, and complex mathematical queries."
    description_zh = "使用 WolframAlpha 求解数学方程和问题。可以处理代数、微积分和复杂的数学查询。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A mathematical question or equation to solve (e.g., 'x^2 + 2x + 1 = 0, what is x?', 'integrate x^2 dx')"
            }
        },
        "required": ["query"]
    }
    
    def __init__(self, cfg=None, use_zh=False, max_retries=3, retry_delay=1):
        """Initialize the Wolfram Alpha tool."""
        super().__init__(cfg, use_zh, max_retries, retry_delay)
        
        if not WOLFRAM_AVAILABLE:
            self.client = None
        else:
            api_key = os.getenv("WOLFRAM_ALPHA_API_KEY")
            if api_key:
                self.client = wolframalpha.Client(api_key)
            else:
                self.client = None
    
    def call(self, params: Union[str, Dict]) -> str:
        """Execute the equation solving operation.
        
        Args:
            params: Parameters containing the query
            
        Returns:
            Solution or error message
        """
        if not WOLFRAM_AVAILABLE:
            return "Error: wolframalpha package is not installed. Please install it with: pip install wolframalpha"
        
        if self.client is None:
            return "Error: WOLFRAM_ALPHA_API_KEY environment variable is not set. Please set it to use this tool."
        
        # Validate and parse parameters
        params_dict = self.verify_json_format_args(params)
        query = params_dict["query"]
        
        try:
            res = self.client.query(query)
            
            if not res.get("@success", False):
                return "Error: Your Wolfram query is invalid. Please try a different query."
            
            # Try to extract the solution
            answer = ""
            for result in res.get("pod", []):
                title = result.get("@title", "")
                if title == "Solution":
                    subpod = result.get("subpod", {})
                    if isinstance(subpod, list):
                        answer = subpod[0].get("plaintext", "")
                    else:
                        answer = subpod.get("plaintext", "")
                    break
                elif title in ["Results", "Solutions"]:
                    subpods = result.get("subpod", [])
                    if not isinstance(subpods, list):
                        subpods = [subpods]
                    for i, sub in enumerate(subpods):
                        answer += f"Solution {i + 1}: {sub.get('plaintext', '')}\n"
                    break
            
            # If no specific solution found, try to get first result
            if not answer:
                try:
                    results = res.results
                    answer = next(results).text
                except (StopIteration, AttributeError):
                    pass
            
            if not answer or answer == "":
                return "Error: No solution found by Wolfram Alpha for this query."
            
            return f"Solution: {answer.strip()}"
            
        except Exception as e:
            return f"Error querying Wolfram Alpha: {str(e)}"

