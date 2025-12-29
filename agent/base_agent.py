import logging
import traceback
import json
import jinja2

from typing import Optional, List, Union, Dict
from abc import ABC, abstractmethod
from tool.base_tool import BasicTool, ModelBasedTool, TOOL_REGISTRY

import tool

logger = logging.getLogger(__name__)


class BasicAgent(ABC):
    """Base class for agents that can use tools to accomplish tasks.
    
    Supports ReAct-style interaction with tools using Action/Action Input/Observation tokens.
    """
    
    def __init__(
            self,
            name: Optional[str] = "",
            description_en: Optional[str] = "",
            description_zh: Optional[str] = "",
            tool_bank: Optional[List[Union[str, Dict]]] = None,
            use_zh: bool = False,
            system_template_dir: str = "./template",
            tool_description_template_en_file: str = "ToolCaller_EN.jinja2",
            tool_description_template_zh_file: str = "ToolCaller_ZH.jinja2",
            special_answer_token: str = "\nAnswer:",
            preload_devices: Optional[List[str]] = None,
    ):
        """Initialize the agent with tools and configuration.
        
        Args:
            name: Agent name
            description_en: English description
            description_zh: Chinese description
            tool_bank: List of tools (str names or dict configs)
            use_zh: Whether to use Chinese language
            system_template_dir: Directory for Jinja2 templates
            tool_description_template_en_file: English tool description template (legacy, kept for compatibility)
            tool_description_template_zh_file: Chinese tool description template (legacy, kept for compatibility)
            special_answer_token: Token to mark final answer (for evaluation)
            preload_devices: Devices for preloading models, e.g., ["cuda:0", "cuda:1"] (default: auto-detect all GPUs)
        """
        self.name = name
        self.description = description_en if not use_zh else description_zh

        self.tool_bank = {}
        
        # Preload tool models before initializing tools
        if tool_bank:
            from tool.model_cache import preload_tools
            tool_names = [t if isinstance(t, str) else t.get("name") for t in tool_bank]
            preload_tools(tool_bank=tool_names, devices=preload_devices)
            
            for t in tool_bank:
                self._init_tool(t)

        self.use_zh = use_zh

        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(system_template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.tool_description_jinja_file = tool_description_template_zh_file if use_zh else tool_description_template_en_file
        
        # Special token for evaluation
        self.special_answer_token = special_answer_token

    def _init_tool(self, tool: Union[str, dict, BasicTool]):
        """Initialize and register a tool to the agent's tool bank.
        
        Model-based tools are automatically loaded to a free GPU.
        
        Args:
            tool: Tool instance, name (str), or config (dict)
        """
        if isinstance(tool, BasicTool):
            instance = tool
            tool_name = tool.name
        else:
            if isinstance(tool, dict):
                tool_name = tool["name"]
                tool_cfg = tool
            elif isinstance(tool, str):
                tool_name = tool
                tool_cfg = None
            else:
                raise ValueError(f"Not supported tool {tool}")

            if tool_name not in TOOL_REGISTRY:
                raise ValueError(f"Tool {tool_name} is not registered!")
            
            instance = TOOL_REGISTRY[tool_name](tool_cfg)
        
        if tool_name in self.tool_bank:
            logger.info(f'Replacing tool {tool_name} in tool bank')
        
        self.tool_bank[tool_name] = instance
        
        # Auto load model-based tools
        if isinstance(instance, ModelBasedTool):
            logger.info(f'Loading {tool_name}...')
            instance.ensure_loaded()
            logger.info(f'{tool_name} loaded on {instance.device}')
    
    def _format_observation(self, result: Dict, tool_name: str = "") -> str:
        """Format tool result dict into human-readable observation.
        
        Simple generic logic: "Key: value" with capitalized first letter.
        Special handling for regions with label and bbox.
        
        Args:
            result: Tool result dict
            tool_name: Tool name for context (optional)
            
        Returns:
            Formatted observation string
        """
        if not result:
            return "success"
        
        # Handle error
        if "error" in result:
            return f"Error: {result['error']}"
        
        # Generic formatting: "Key: value"
        parts = []
        for key, value in result.items():
            # Capitalize first letter of key
            formatted_key = key[0].upper() + key[1:] if key else key
            
            # Format value
            if isinstance(value, list):
                # Special handling for regions (list of dicts with label and bbox)
                if value and isinstance(value[0], dict) and "label" in value[0] and "bbox" in value[0]:
                    region_strs = []
                    for item in value:
                        label = item["label"]
                        bbox = item["bbox"]
                        region_strs.append(f"{label} at {bbox}")
                    formatted_value = ", ".join(region_strs)
                else:
                    formatted_value = str(value)
            else:
                formatted_value = str(value)
            
            parts.append(f"{formatted_key}: {formatted_value}")
        
        return ", ".join(parts)

    def _call_tool(
            self,
            tool_name: str,
            tool_args: Union[str, dict] = "{}",
            **kwargs
    ):
        """Call a tool with given arguments and handle errors.
        
        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool (JSON string or dict)
            **kwargs: Additional keyword arguments
            
        Returns:
            Tool result as string or JSON-formatted string
        """
        if tool_name not in self.tool_bank:
            return f"Tool {tool_name} does not exist!"
        tool = self.tool_bank[tool_name]
        try:
            tool_result = tool.call(tool_args, **kwargs)
        except Exception as ex:
            exception_type = type(ex).__name__
            exception_message = str(ex)
            traceback_info = ''.join(traceback.format_tb(ex.__traceback__))
            error_message = f'An error occurred when calling tool `{tool_name}`:\n' \
                            f'{exception_type}: {exception_message}\n' \
                            f'Traceback:\n{traceback_info}'
            logging.info(error_message)
            return error_message

        # If result is a dict (new format with PIL.Image), return as is
        if isinstance(tool_result, dict):
            return tool_result
        
        # If result is a string (legacy format), return as is
        if isinstance(tool_result, str):
            return tool_result
        
        # Otherwise, try to serialize
        try:
            return json.dumps(tool_result, ensure_ascii=False, indent=4)
        except (TypeError, ValueError):
            # Can't serialize (e.g., contains PIL.Image), return as is
            return tool_result


    def _build_tools_schema(self) -> List[Dict]:
        """Build OpenAI-format tools schema from registered tools.
        
        Returns:
            List of tool definitions in OpenAI format
        """
        tools = []
        for tool in self.tool_bank.values():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name_for_model,
                    "description": tool.tool_description,
                    "parameters": tool.parameters
                }
            })
        return tools
    

    def cleanup(self) -> None:
        """Cleanup all model-based tools and release GPU resources."""
        for name, t in self.tool_bank.items():
            if isinstance(t, ModelBasedTool) and t.is_loaded:
                logger.info(f'Unloading {name} from {t.device}')
                t.unload_model()
    
    def __del__(self):
        """Auto cleanup on deletion."""
        try:
            self.cleanup()
        except Exception as e:
            logger.debug(f"Cleanup failed during deletion: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.cleanup()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        self.cleanup()

    @abstractmethod
    async def act(
            self,
            **kwargs
    ):
        """Execute agent action. Must be implemented by subclasses.
        
        Args:
            **kwargs: Task-specific arguments
            
        Returns:
            Agent action result
        """
        raise NotImplementedError