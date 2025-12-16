"""Tool for visualizing regions on images with bounding boxes."""

from typing import Union, Dict
from tool.base_tool import BasicTool, register_tool

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@register_tool(name="visualize_regions")
class VisualizeRegionsOnImageTool(BasicTool):
    """A tool to label regions on an image with bounding boxes."""
    
    name = "visualize_regions"
    description_en = "Label regions on an image with bounding boxes and optional text labels. Each region is defined by normalized coordinates [x1, y1, x2, y2] where values are between 0 and 1."
    description_zh = "在图像上标注区域并绘制边界框和可选的文本标签。每个区域由归一化坐标 [x1, y1, x2, y2] 定义，其中值在 0 到 1 之间。"
    parameters = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the input image file"
            },
            "regions": {
                "type": "array",
                "description": "List of regions to label, each with 'bbox' and optional 'label'",
                "items": {
                    "type": "object",
                    "properties": {
                        "bbox": {
                            "type": "array",
                            "description": "Normalized bounding box [x1, y1, x2, y2]",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4
                        },
                        "label": {
                            "type": "string",
                            "description": "Optional label text for the region"
                        }
                    },
                    "required": ["bbox"]
                }
            },
            "color": {
                "type": "string",
                "description": "Color of the bounding boxes (default: 'yellow')"
            },
            "width": {
                "type": "integer",
                "description": "Width of the bounding box lines (default: 4)"
            }
        },
        "required": ["image_path", "regions"]
    }
    example = '{"image_path": "/path/to/image.jpg", "regions": [{"bbox": [0.1, 0.1, 0.5, 0.5], "label": "Object A"}, {"bbox": [0.6, 0.6, 0.9, 0.9], "label": "Object B"}]}'
    
    def call(self, params: Union[str, Dict]) -> str:
        """Execute the visualization operation.
        
        Args:
            params: Parameters containing image_path, regions, and optional styling
            
        Returns:
            PIL.Image object with visualized regions
        """
        if not PIL_AVAILABLE:
            return "Error: PIL (Pillow) is not installed. Please install it with: pip install Pillow"
        
        # Validate and parse parameters
        params_dict = self.parse_params(params)
        
        image_path = params_dict["image_path"]
        regions = params_dict["regions"]
        color = params_dict.get("color", "yellow")
        width = params_dict.get("width", 4)
        
        try:
            # Load image
            image = Image.open(image_path)
            W, H = image.size
            
            # Create a copy to draw on
            img_labeled = image.copy()
            draw = ImageDraw.Draw(img_labeled)
            
            # Try to load a font, fall back to default if not available
            try:
                font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 16)
            except (OSError, IOError):
                try:
                    # Try common macOS font path
                    font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 16)
                except (OSError, IOError):
                    # Fall back to default font
                    font = ImageFont.load_default()
            
            text_color = 'black'
            
            # Draw each region
            for obj in regions:
                bbox = obj['bbox']
                # Convert normalized coordinates to pixel coordinates
                bbox_pixels = (bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H)
                
                # Draw rectangle
                draw.rectangle(bbox_pixels, outline=color, width=width)
                
                x1, y1, x2, y2 = bbox_pixels
                label = obj.get('label', '')
                
                # Draw label if present
                if label:
                    # Get text size
                    try:
                        bbox_text = draw.textbbox((0, 0), label, font=font)
                        w = bbox_text[2] - bbox_text[0]
                        h = bbox_text[3] - bbox_text[1]
                    except AttributeError:
                        # Fallback for older Pillow versions
                        w, h = font.getsize(label)
                    
                    # Position label at bottom of bbox, or top if it would go outside image
                    if x1 + w > W or y2 + h > H:
                        draw.rectangle((x1, y2 - h, x1 + w, y2), fill=color)
                        draw.text((x1, y2 - h), label, fill=text_color, font=font)
                    else:
                        draw.rectangle((x1, y2, x1 + w, y2 + h), fill=color)
                        draw.text((x1, y2), label, fill=text_color, font=font)
            
            # Return dict with PIL Image object
            return {"output_image": img_labeled}
            
        except FileNotFoundError:
            return {"error": f"Image file not found at {image_path}"}
        except Exception as e:
            return {"error": str(e)}
    
    def generate_description(self, properties, observation):
        """Generate description for visualized regions."""
        img = properties.get("image_path", "image")
        regions = properties.get("regions", [])
        num_regions = len(regions) if isinstance(regions, list) else 0
        return f"Visualized {num_regions} regions on {img}"

