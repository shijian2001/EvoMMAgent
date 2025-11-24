"""Tool for visualizing regions on images with bounding boxes."""

import os
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
                "description": "List of regions to label. Each region should have 'bbox' (list of 4 floats) and optional 'label' (string)",
                "items": {
                    "type": "object",
                    "properties": {
                        "bbox": {
                            "type": "array",
                            "description": "Normalized bounding box coordinates [x1, y1, x2, y2]",
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
                "description": "Color of the bounding boxes (default: 'yellow')",
                "default": "yellow"
            },
            "width": {
                "type": "integer",
                "description": "Width of the bounding box lines (default: 4)",
                "default": 4
            },
            "output_path": {
                "type": "string",
                "description": "Path to save the output image (optional, if not provided, will save as 'image_path_labeled.png')"
            }
        },
        "required": ["image_path", "regions"]
    }
    
    def call(self, params: Union[str, Dict]) -> str:
        """Execute the visualization operation.
        
        Args:
            params: Parameters containing image_path, regions, and optional styling
            
        Returns:
            Path to the output image or error message
        """
        if not PIL_AVAILABLE:
            return "Error: PIL (Pillow) is not installed. Please install it with: pip install Pillow"
        
        # Validate and parse parameters
        params_dict = self.verify_json_format_args(params)
        
        image_path = params_dict["image_path"]
        regions = params_dict["regions"]
        color = params_dict.get("color", "yellow")
        width = params_dict.get("width", 4)
        output_path = params_dict.get("output_path")
        
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
            except:
                try:
                    # Try common macOS font path
                    font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 16)
                except:
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
            
            # Save the output
            if not output_path:
                base, ext = os.path.splitext(image_path)
                output_path = f"{base}_labeled.png"
            
            img_labeled.save(output_path)
            
            return f"Image saved to: {output_path}"
            
        except FileNotFoundError:
            return f"Error: Image file not found at {image_path}"
        except Exception as e:
            return f"Error: {str(e)}"

