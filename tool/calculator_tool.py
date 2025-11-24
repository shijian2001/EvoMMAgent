"""Example calculator tool for basic arithmetic operations."""

from typing import Union, Dict
from tool.base_tool import BasicTool, register_tool


@register_tool(name="calculator")
class CalculatorTool(BasicTool):
    """A simple calculator tool that performs basic arithmetic operations."""
    
    name = "calculator"
    description_en = "Performs basic arithmetic operations (add, subtract, multiply, divide)"
    description_zh = "执行基本算术运算（加、减、乘、除）"
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The arithmetic operation to perform"
            },
            "a": {
                "type": "number",
                "description": "First operand"
            },
            "b": {
                "type": "number",
                "description": "Second operand"
            }
        },
        "required": ["operation", "a", "b"]
    }
    
    def call(self, params: Union[str, Dict]) -> str:
        """Execute the calculator operation.
        
        Args:
            params: Parameters containing operation, a, and b
            
        Returns:
            Calculation result as string
        """
        # Validate and parse parameters
        params_dict = self.verify_json_format_args(params)
        
        operation = params_dict["operation"]
        a = params_dict["a"]
        b = params_dict["b"]
        
        # Perform calculation
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return "Error: Division by zero"
            result = a / b
        else:
            return f"Error: Unknown operation '{operation}'"
        
        return f"Result: {result}"

