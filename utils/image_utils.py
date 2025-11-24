"""Image processing utility functions."""

import os
from PIL import Image, ImageDraw


# Default input image path (can be configured)
INPUT_IMAGE_PATH = os.environ.get("INPUT_IMAGE_PATH", "./images")


def get_full_path_data(full_filename: str) -> str:
    """Get the full path of an image file.
    
    Args:
        full_filename: A string representing the filename
        
    Returns:
        Full path to the image file or None if not found
    """
    extensions = [".png", ".webp", ".jpg"]
    filename, curr_extension = os.path.splitext(full_filename)
    
    if full_filename.find("/") == -1:  # Try adding the image base path
        base_path = INPUT_IMAGE_PATH
        img_path = os.path.join(base_path, filename)
        if os.path.exists(img_path):
            return img_path
    else:
        # Try other image file extensions in the same directory
        for ext in extensions:
            if ext == curr_extension:
                continue
            new_filename = full_filename.replace(curr_extension, ext)
            if os.path.exists(new_filename):
                return new_filename
    
    return None


def image_processing(img, return_path: bool = False):
    """Process image input - convert to PIL Image or return path.
    
    Args:
        img: A string representing the image file path or an image object
        return_path: Whether to return the path of the image file
        
    Returns:
        PIL Image object in RGB format or path string
        
    Raises:
        FileNotFoundError: If image file is not found
    """
    if isinstance(img, Image.Image):
        assert return_path == False, "Cannot return path for an image object input"
        return img.convert("RGB")
    elif isinstance(img, str):
        final_path = img
        if not os.path.exists(img):
            final_path = get_full_path_data(img)
        if final_path:
            return final_path if return_path else Image.open(final_path).convert("RGB")
        else:
            raise FileNotFoundError(f"Image file not found: {img}")


def expand_bbox(bbox: tuple, original_image_size: tuple, margin: float = 0.5) -> tuple:
    """Expand bounding box by margin around its center.
    
    Args:
        bbox: A tuple (left, top, right, bottom)
        original_image_size: A tuple (width, height) of the original image size
        margin: Expansion margin (ratio if <=1.0, absolute pixels if >1.0)
        
    Returns:
        A tuple (new_left, new_top, new_right, new_bottom)
    """
    left, upper, right, lower = bbox
    width = right - left
    height = lower - upper
    
    # Calculate the new width and height
    new_width = width * (1 + margin) if margin <= 1.0 else width + margin
    new_height = height * (1 + margin) if margin <= 1.0 else height + margin
    
    # Calculate the center of the original bounding box
    center_x = left + width / 2
    center_y = upper + height / 2
    
    # Determine the new bounding box coordinates
    new_left = max(0, center_x - new_width / 2)
    new_upper = max(0, center_y - new_height / 2)
    new_right = min(original_image_size[0], center_x + new_width / 2)
    new_lower = min(original_image_size[1], center_y + new_height / 2)
    
    return (int(new_left), int(new_upper), int(new_right), int(new_lower))


def visualize_bbox(image: Image.Image, bbox: list, color: str = "red", width: int = 3) -> Image.Image:
    """Visualize bounding box on image.
    
    Args:
        image: PIL Image
        bbox: [left, top, right, bottom] in percentage (0-1)
        color: Box color
        width: Line width
        
    Returns:
        Image with bbox visualization
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    W, H = image.size
    pixel_bbox = [bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H]
    
    # Draw rectangle
    draw.rectangle(pixel_bbox, outline=color, width=width)
    
    return img_copy

