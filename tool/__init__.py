"""Tool modules for EvoMMAgent."""

from tool.base_tool import BasicTool, ModelBasedTool, TOOL_REGISTRY, register_tool
# GPU management removed - use preload_tools() for GPU allocation

# Import tools to trigger registration
from tool.calculator_tool import CalculatorTool
from tool.crop_tool import CropTool
from tool.zoom_in_tool import ZoomInTool
from tool.visualize_regions_tool import VisualizeRegionsOnImageTool
from tool.solve_math_equation_tool import SolveMathEquationTool
from tool.get_objects_tool import GetObjectsTool
from tool.localize_objects_tool import LocalizeObjectsTool
from tool.detect_faces_tool import DetectFacesTool
from tool.estimate_region_depth_tool import EstimateRegionDepthTool
from tool.estimate_object_depth_tool import EstimateObjectDepthTool
from tool.get_image2images_similarity_tool import GetImageToImagesSimilarityTool
from tool.get_image2texts_similarity_tool import GetImageToTextsSimilarityTool
from tool.get_text2images_similarity_tool import GetTextToImagesSimilarityTool
from tool.ocr_tool import OCRTool

__all__ = [
    "BasicTool",
    "ModelBasedTool", 
    "TOOL_REGISTRY",
    "register_tool",
]
