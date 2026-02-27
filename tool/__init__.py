"""Tool modules for EvoMMAgent."""

from tool.base_tool import BasicTool, ModelBasedTool, TOOL_REGISTRY, register_tool
# GPU management removed - use preload_tools() for GPU allocation

# Import tools to trigger registration.
# In lightweight test environments, some optional deps (e.g. torch) may be missing.
try:
    from tool.calculator_tool import CalculatorTool
except Exception:
    pass
try:
    from tool.crop_tool import CropTool
except Exception:
    pass
try:
    from tool.zoom_in_tool import ZoomInTool
except Exception:
    pass
try:
    from tool.visualize_regions_tool import VisualizeRegionsOnImageTool
except Exception:
    pass
try:
    from tool.solve_math_equation_tool import SolveMathEquationTool
except Exception:
    pass
try:
    from tool.get_objects_tool import GetObjectsTool
except Exception:
    pass
try:
    from tool.localize_objects_tool import LocalizeObjectsTool
except Exception:
    pass
try:
    from tool.detect_faces_tool import DetectFacesTool
except Exception:
    pass
try:
    from tool.estimate_region_depth_tool import EstimateRegionDepthTool
except Exception:
    pass
try:
    from tool.estimate_object_depth_tool import EstimateObjectDepthTool
except Exception:
    pass
try:
    from tool.get_image2images_similarity_tool import GetImageToImagesSimilarityTool
except Exception:
    pass
try:
    from tool.get_image2texts_similarity_tool import GetImageToTextsSimilarityTool
except Exception:
    pass
try:
    from tool.get_text2images_similarity_tool import GetTextToImagesSimilarityTool
except Exception:
    pass
try:
    from tool.ocr_tool import OCRTool
except Exception:
    pass
try:
    from tool.get_images_tool import GetImagesTool
except Exception:
    pass
try:
    from tool.measure_region_property_tool import MeasureRegionPropertyTool
except Exception:
    pass
try:
    from tool.search_experiences_tool import SearchExperiencesTool
except Exception:
    pass

__all__ = [
    "BasicTool",
    "ModelBasedTool", 
    "TOOL_REGISTRY",
    "register_tool",
]
