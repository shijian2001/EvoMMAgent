"""Task-level memory management with multimodal outputs and traces."""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from threading import Lock

logger = logging.getLogger(__name__)


class Memory:
    """Task-level memory management with multimodal outputs and traces."""
    
    # Global task counter for concurrent tasks
    _task_counter = 0
    _counter_lock = Lock()
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize memory manager.
        
        Args:
            base_dir: Base directory for memory storage (uses config default if None)
        """
        if base_dir is None:
            try:
                from config import Config
                base_dir = Config.default().agent.memory_dir
            except (ImportError, AttributeError):
                base_dir = "memory"
        
        self.base_dir = base_dir
        self.tasks_dir = os.path.join(base_dir, "tasks")
        os.makedirs(self.tasks_dir, exist_ok=True)
        
        self._initialize_counter_from_disk()
        
        self.task_id = None
        self.task_dir = None
        self.trace_data = None
        self.ref_counters = {"img": 0, "vid": 0}
        self._path_to_id = {}
    
    def _initialize_counter_from_disk(self):
        """Initialize task counter from existing task directories."""
        with Memory._counter_lock:
            if Memory._task_counter == 0:
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
            task_id = cls._task_counter
            cls._task_counter += 1
            return f"{task_id:06d}"
    
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
        self.ref_counters = {"img": 0, "vid": 0}
        self._path_to_id = {}
        
        return self.task_id
    
    def add_input(self, file_path: str, modality: str = "img") -> str:
        """Add input file to task.
        
        Args:
            file_path: Path to input file
            modality: Type of input (img, vid)
            
        Returns:
            id: Reference ID for the input (e.g., "img_0")
        """
        id_str = f"{modality}_{self.ref_counters[modality]}"
        self.ref_counters[modality] += 1
        
        ext = os.path.splitext(file_path)[1]
        dst_path = os.path.join(self.task_dir, f"{id_str}{ext}")
        shutil.copy2(file_path, dst_path)
        
        self._path_to_id[file_path] = id_str
        self._path_to_id[dst_path] = id_str
        
        modality_map = {"img": "images", "vid": "videos"}
        modality_key = modality_map.get(modality, f"{modality}s")
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
        output_type: str = "img",
        description: Optional[str] = None
    ) -> Optional[str]:
        """Log an action step.
        
        Args:
            tool: Tool name
            properties: Tool input parameters (e.g., {"image": "img_0", "bbox": [...]})
            observation: Tool output - string if no multimodal output, dict with tool data otherwise
            output_object: Optional multimodal object (PIL.Image, video path, etc.)
            output_type: Type of output (img, vid)
            description: Optional pre-generated description for multimodal output
            
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
            if description is None:
                description = f"{tool} output"
            
            observation_dict = {
                "id": output_id,
                "type": modality,
                "description": description
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
            modality: Type (img, vid)
        """
        try:
            dst_path = None
            
            if modality == "img":
                # Handle PIL Image
                try:
                    from PIL import Image
                    if isinstance(obj, Image.Image):
                        dst_path = os.path.join(self.task_dir, f"{output_id}.png")
                        obj.save(dst_path)
                    elif isinstance(obj, str) and os.path.exists(obj):
                        ext = os.path.splitext(obj)[1]
                        dst_path = os.path.join(self.task_dir, f"{output_id}{ext}")
                        shutil.copy2(obj, dst_path)
                        # Record original path mapping
                        self._path_to_id[obj] = output_id
                except ImportError:
                    pass
            
            elif modality == "vid":
                # Handle video file path
                if isinstance(obj, str) and os.path.exists(obj):
                    ext = os.path.splitext(obj)[1]
                    dst_path = os.path.join(self.task_dir, f"{output_id}{ext}")
                    shutil.copy2(obj, dst_path)
                    # Record original path mapping
                    self._path_to_id[obj] = output_id
            
            # Fallback: try to copy as file
            else:
                if isinstance(obj, str) and os.path.exists(obj):
                    ext = os.path.splitext(obj)[1] or ".dat"
                    dst_path = os.path.join(self.task_dir, f"{output_id}{ext}")
                    shutil.copy2(obj, dst_path)
                    # Record original path mapping
                    self._path_to_id[obj] = output_id
            
            # Record task_dir path mapping
            if dst_path:
                self._path_to_id[dst_path] = output_id
                
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
            if isinstance(value, str) and (value.startswith("img_") or value.startswith("vid_")):
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
    
    def resolve_paths_to_ids(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve file paths back to reference IDs in observation data.
        
        Uses pre-maintained mapping table for O(1) lookup.
        
        Args:
            data: Observation data that may contain file paths
            
        Returns:
            Data with file paths replaced by reference IDs
            
        Example:
            Input:  {"best match": "/path/to/img.png"}
            Output: {"best match": "img_1"}
        """
        def resolve(value):
            """Recursively replace paths with IDs."""
            if isinstance(value, str):
                # Direct lookup in mapping table - O(1)
                return self._path_to_id.get(value, value)
            
            if isinstance(value, dict):
                # Recursively handle dict
                return {k: resolve(v) for k, v in value.items()}
            
            if isinstance(value, list):
                # Recursively handle list
                return [resolve(item) for item in value]
            
            # Other types return as-is
            return value
        
        return resolve(data)
