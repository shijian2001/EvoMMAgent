"""Task-level memory management with multimodal outputs and traces."""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from threading import Lock


class Memory:
    """Task-level memory management with multimodal outputs and traces."""
    
    # Global task counter for concurrent tasks
    _task_counter = 0
    _counter_lock = Lock()
    
    def __init__(self, base_dir: str = "memory"):
        """Initialize memory manager.
        
        Args:
            base_dir: Base directory for memory storage
        """
        self.base_dir = base_dir
        self.tasks_dir = os.path.join(base_dir, "tasks")
        os.makedirs(self.tasks_dir, exist_ok=True)
        
        # Initialize counter from existing tasks (only once)
        self._initialize_counter_from_disk()
        
        # Current task state
        self.task_id = None
        self.task_dir = None
        self.trace_data = None
        self.ref_counters = {"img": 0, "vid": 0, "audio": 0}
    
    def _initialize_counter_from_disk(self):
        """Initialize task counter from existing task directories."""
        with Memory._counter_lock:
            if Memory._task_counter == 0:
                # Check existing task directories
                if os.path.exists(self.tasks_dir):
                    existing = [d for d in os.listdir(self.tasks_dir) 
                               if os.path.isdir(os.path.join(self.tasks_dir, d)) and d.isdigit()]
                    if existing:
                        max_id = max(int(d) for d in existing)
                        Memory._task_counter = max_id
    
    @classmethod
    def _get_next_task_id(cls) -> str:
        """Thread-safe task ID generation."""
        with cls._counter_lock:
            cls._task_counter += 1
            return f"{cls._task_counter:06d}"
    
    def start_task(self, question: str) -> str:
        """Start a new task.
        
        Args:
            question: User query
            
        Returns:
            task_id: Unique task ID
        """
        self.task_id = self._get_next_task_id()
        self.task_dir = os.path.join(self.tasks_dir, self.task_id)
        os.makedirs(self.task_dir, exist_ok=True)
        
        self.trace_data = {
            "task_id": self.task_id,
            "input": {
                "question": question
            },
            "trace": []
        }
        self.ref_counters = {"img": 0, "vid": 0, "audio": 0}
        
        return self.task_id
    
    def add_input(self, file_path: str, modality: str = "img") -> str:
        """Add input file to task.
        
        Args:
            file_path: Path to input file
            modality: Type of input (img, vid, audio)
            
        Returns:
            id: Reference ID for the input (e.g., "img_0")
        """
        id_str = f"{modality}_{self.ref_counters[modality]}"
        self.ref_counters[modality] += 1
        
        # Copy file to task dir with id as name
        ext = os.path.splitext(file_path)[1]
        dst_path = os.path.join(self.task_dir, f"{id_str}{ext}")
        shutil.copy2(file_path, dst_path)
        
        # Add to input dict with original path
        modality_key = f"{modality}s" if modality == "img" else f"{modality}s"  # images, videos, audios
        if modality_key not in self.trace_data["input"]:
            self.trace_data["input"][modality_key] = []
        
        self.trace_data["input"][modality_key].append({
            "id": id_str,
            "path": file_path
        })
        
        return id_str
    
    def log_think(self, content: str, special_think_token: str = "Thought:"):
        """Log a thinking step.
        
        Args:
            content: Thought content
            special_think_token: Special token to remove (default "Thought:")
        """
        # Clean up content: remove special_think_token prefix and strip
        content = content.strip()
        if content.startswith(special_think_token):
            content = content[len(special_think_token):].strip()
        
        step = len(self.trace_data["trace"]) + 1
        self.trace_data["trace"].append({
            "step": step,
            "type": "think",
            "content": content
        })
    
    def log_action(
        self,
        tool: str,
        properties: Dict[str, Any],
        observation: Union[str, Dict[str, Any]],
        output_object: Optional[Any] = None,
        output_type: str = "img"
    ) -> Optional[str]:
        """Log an action step.
        
        Args:
            tool: Tool name
            properties: Tool input parameters (e.g., {"image": "img_0", "bbox": [...]})
            observation: Tool output - string if no multimodal output, dict with tool data otherwise
            output_object: Optional multimodal object (PIL.Image, video path, etc.)
            output_type: Type of output (img, vid, audio)
            
        Returns:
            output_id: Reference ID of output if created, else None
        """
        step = len(self.trace_data["trace"]) + 1
        output_id = None
        
        # If tool produces multimodal output
        if output_object is not None:
            modality = output_type
            output_id = f"{modality}_{self.ref_counters[modality]}"
            self.ref_counters[modality] += 1
            
            # Save object to file
            self._save_output_object(output_object, output_id, modality)
            
            # Build observation dict with id and description
            observation_dict = {
                "id": output_id,
                "type": modality,
                "description": self._generate_description(tool, properties, observation)
            }
            
            # Add any additional data from original observation
            if isinstance(observation, dict):
                observation_dict.update(observation)
            
            observation = observation_dict
        
        # Append to trace
        self.trace_data["trace"].append({
            "step": step,
            "type": "action",
            "tool": tool,
            "properties": properties,
            "observation": observation
        })
        
        return output_id
    
    def _save_output_object(self, obj: Any, output_id: str, modality: str):
        """Save multimodal object to file.
        
        Args:
            obj: Multimodal object (PIL.Image, file path, etc.)
            output_id: Reference ID
            modality: Type (img, vid, audio)
        """
        try:
            if modality == "img":
                # Handle PIL Image
                try:
                    from PIL import Image
                    if isinstance(obj, Image.Image):
                        dst_path = os.path.join(self.task_dir, f"{output_id}.png")
                        obj.save(dst_path)
                        return
                except ImportError:
                    pass
                
                # Handle file path
                if isinstance(obj, str) and os.path.exists(obj):
                    ext = os.path.splitext(obj)[1]
                    dst_path = os.path.join(self.task_dir, f"{output_id}{ext}")
                    shutil.copy2(obj, dst_path)
                    return
            
            elif modality == "vid":
                # Handle video file path
                if isinstance(obj, str) and os.path.exists(obj):
                    ext = os.path.splitext(obj)[1]
                    dst_path = os.path.join(self.task_dir, f"{output_id}{ext}")
                    shutil.copy2(obj, dst_path)
                    return
            
            # Fallback: try to copy as file
            if isinstance(obj, str) and os.path.exists(obj):
                ext = os.path.splitext(obj)[1] or ".dat"
                dst_path = os.path.join(self.task_dir, f"{output_id}{ext}")
                shutil.copy2(obj, dst_path)
        except Exception as e:
            import logging
            logging.error(f"Failed to save output object {output_id}: {e}")
    
    def log_answer(self, content: str):
        """Log final answer.
        
        Args:
            content: Answer content
        """
        step = len(self.trace_data["trace"]) + 1
        self.trace_data["trace"].append({
            "step": step,
            "type": "answer",
            "content": content
        })
        self.trace_data["answer"] = content
    
    def end_task(self, success: bool = True):
        """Finalize task and save trace.
        
        Args:
            success: Whether task completed successfully
        """
        self.trace_data["success"] = success
        
        # Save trace.json
        trace_path = os.path.join(self.task_dir, "trace.json")
        with open(trace_path, 'w', encoding='utf-8') as f:
            json.dump(self.trace_data, f, indent=2, ensure_ascii=False)
    
    def get_file_path(self, ref_id: str) -> Optional[str]:
        """Get file path from reference ID.
        
        Args:
            ref_id: Reference ID (e.g., "img_0")
            
        Returns:
            Full file path, or None if not found
        """
        if not self.task_dir:
            return None
        
        # Check in task directory for files matching the ref_id
        import glob
        pattern = os.path.join(self.task_dir, f"{ref_id}.*")
        matches = glob.glob(pattern)
        return matches[0] if matches else None
    
    def resolve_ids(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve reference IDs to file paths in properties.
        
        Args:
            properties: Tool properties that may contain IDs
            
        Returns:
            Properties with IDs replaced by file paths
        """
        def resolve_value(value):
            if isinstance(value, str) and (value.startswith("img_") or value.startswith("vid_") or value.startswith("audio_")):
                # Try to resolve as ID
                path = self.get_file_path(value)
                return path if path else value
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(v) for v in value]
            else:
                return value
        
        return {key: resolve_value(value) for key, value in properties.items()}
    
    def _generate_description(
        self, 
        tool: str, 
        properties: Dict[str, Any],
        observation: Union[str, Dict[str, Any]]
    ) -> str:
        """Generate description from template rules.
        
        Args:
            tool: Tool name
            properties: Tool input parameters
            observation: Tool output data
            
        Returns:
            description: Human-readable description
        """
        observation_data = observation if isinstance(observation, dict) else {}
        
        rules = {
            "segment": lambda p, o: (
                f"Segmentation of {p.get('image')} "
                f"with {len(o.get('bboxes', []))} objects"
            ),
            "crop": lambda p, o: (
                f"Cropped {p.get('image')} at bbox {p.get('bbox')}"
            ),
            "zoom_in": lambda p, o: (
                f"Zoomed {p.get('image')} at {p.get('bbox')} "
                f"by {p.get('factor', 2)}x"
            ),
            "ocr": lambda p, o: (
                f"OCR from {p.get('image')}"
            ),
            "get_objects": lambda p, o: (
                f"Objects detected in {p.get('image')}"
            ),
            "localize_objects": lambda p, o: (
                f"Localized '{p.get('query')}' in {p.get('image')}"
            ),
            "estimate_depth": lambda p, o: (
                f"Depth estimation of {p.get('image')}"
            ),
            "estimate_region_depth": lambda p, o: (
                f"Depth estimation of {p.get('image')} at {p.get('bbox')}"
            ),
            "detect_faces": lambda p, o: (
                f"Face detection in {p.get('image')}"
            ),
            "visualize_regions": lambda p, o: (
                f"Visualized regions in {p.get('image')}"
            ),
            "calculator": lambda p, o: (
                f"Calculated: {p.get('expression')}"
            ),
        }
        
        rule_fn = rules.get(tool, lambda p, o: f"{tool} output")
        try:
            return rule_fn(properties, observation_data)
        except Exception:
            return f"{tool} output"
