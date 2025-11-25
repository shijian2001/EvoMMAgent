"""Tool modules for EvoMMAgent."""

# Import all tools to register them
from tool.calculator_tool import CalculatorTool
from tool.zoom_in_tool import ZoomInTool
from tool.crop_tool import CropTool
from tool.visualize_regions_tool import VisualizeRegionsOnImageTool
from tool.solve_math_equation_tool import SolveMathEquationTool

__all__ = [
    "CalculatorTool",
    "ZoomInTool",
    "CropTool",
    "VisualizeRegionsTool",
    "SolveMathEquationTool",
]

