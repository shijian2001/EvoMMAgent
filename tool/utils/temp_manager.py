"""Temporary file management for tools."""

import os
import tempfile
from typing import Optional
from datetime import datetime


class TempManager:
    """Manage temporary files for tools in a centralized location."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize temp manager.
        
        Args:
            base_dir: Base directory for temp files. If None, uses system temp directory.
        """
        if base_dir is None:
            base_dir = os.path.join(tempfile.gettempdir(), "evommagent_outputs")
        
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
    
    def get_output_path(
        self,
        tool_name: str,
        input_path: Optional[str] = None,
        suffix: str = "",
        extension: str = None,
    ) -> str:
        """Get output file path for a tool.
        
        Args:
            tool_name: Name of the tool (e.g., "zoom_in", "crop")
            input_path: Optional input file path to derive name from
            suffix: Optional suffix to add to filename
            extension: File extension (e.g., ".jpg", ".mp4"). If None, infers from input_path
            
        Returns:
            Full output file path
        """
        # Create tool-specific subdirectory
        tool_dir = os.path.join(self.base_dir, tool_name)
        os.makedirs(tool_dir, exist_ok=True)
        
        # Generate filename
        if input_path:
            basename = os.path.basename(input_path)
            name, ext = os.path.splitext(basename)
            if extension is None:
                extension = ext
        else:
            # Use timestamp if no input path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f"output_{timestamp}"
            if extension is None:
                extension = ".jpg"
        
        # Combine components
        if suffix:
            filename = f"{name}_{suffix}{extension}"
        else:
            filename = f"{name}{extension}"
        
        return os.path.join(tool_dir, filename)
    
    def cleanup(self, tool_name: Optional[str] = None, max_age_hours: int = 24):
        """Clean up old output files.
        
        Args:
            tool_name: Optional specific tool to clean. If None, cleans all.
            max_age_hours: Maximum age in hours before deletion
        """
        import time
        
        if tool_name:
            dirs_to_clean = [os.path.join(self.base_dir, tool_name)]
        else:
            dirs_to_clean = [
                os.path.join(self.base_dir, d)
                for d in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, d))
            ]
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for dir_path in dirs_to_clean:
            if not os.path.exists(dir_path):
                continue
            
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass


# Global temp manager instance
_global_temp_manager = None


def get_temp_manager() -> TempManager:
    """Get the global temp manager instance.
    
    Returns:
        Global TempManager instance
    """
    global _global_temp_manager
    if _global_temp_manager is None:
        _global_temp_manager = TempManager()
    return _global_temp_manager

