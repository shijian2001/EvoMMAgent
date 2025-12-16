"""Calculator tool for evaluating mathematical expressions."""

import re
from typing import Union, Dict
from tool.base_tool import BasicTool, register_tool


@register_tool(name="calculator")
class CalculatorTool(BasicTool):
    """An enhanced calculator tool that evaluates mathematical expressions."""
    
    name = "calculator"
    description_en = "Calculate mathematical expressions. Supports basic arithmetic (+, -, *, /), exponentiation (**), and parentheses."
    description_zh = "计算数学表达式。支持基本算术运算（+、-、*、/）、幂运算（**）和括号。"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to calculate"
            }
        },
        "required": ["expression"]
    }
    example = '{"expression": "123 * 456 + 789"}'
    
    def call(self, params: Union[str, Dict]) -> Dict:
        """Execute calculator operation.
        
        Args:
            params: Parameters containing the expression
            
        Returns:
            Dict with result or error
        """
        # Validate and parse parameters
        params_dict = self.parse_params(params)
        expression = params_dict["expression"]
        
        # Sanitize the expression - only allow safe mathematical operations
        if not self._is_safe_expression(expression):
            return {"error": "Expression contains invalid or unsafe characters"}
        
        try:
            # Use eval with restricted namespace for safety
            result = eval(expression, {"__builtins__": {}}, {})
            return {"calculation result": result}
        except ZeroDivisionError:
            return {"error": "Division by zero"}
        except SyntaxError:
            return {"error": "Invalid syntax in expression"}
        except Exception as e:
            return {"error": str(e)}
    
    def _is_safe_expression(self, expr: str) -> bool:
        """Check if expression contains only safe mathematical operations.
        
        Args:
            expr: Expression to validate
            
        Returns:
            True if expression is safe, False otherwise
        """
        # Allow only digits, operators, parentheses, decimal points, and whitespace
        safe_pattern = re.compile(r'^[\d\s\+\-\*\/\(\)\.\*\*]+$')
        return bool(safe_pattern.match(expr))

