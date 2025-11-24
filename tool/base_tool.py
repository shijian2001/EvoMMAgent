import json
import logging
import jsonschema

from abc import ABC, abstractmethod
from typing import Union, List, Optional, Dict

TOOL_REGISTRY = {}


def register_tool(
        name: str,
        allow_overwrite: bool = False,
):
    """Decorator to register a tool class to the global registry.
    
    Args:
        name: Unique name for the tool
        allow_overwrite: Whether to allow overwriting existing tool
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        if name in TOOL_REGISTRY:
            if allow_overwrite:
                logging.info(f'Tool `{name}` already exists! Overwriting with class {cls}.')
            else:
                raise ValueError(f'Tool `{name}` already exists! Please ensure that the tool name is unique.')
        if cls.name and (cls.name != name):
            raise ValueError(f'{cls.__name__}.name="{cls.name}" conflicts with @register_tool(name="{name}").')
        cls.name = name
        TOOL_REGISTRY[name] = cls

        return cls

    return decorator


class BasicTool(ABC):
    """Base class for all tools that can be used by agents.
    
    Attributes:
        name: Tool identifier
        description_en: English description of the tool
        description_zh: Chinese description of the tool
        parameters: Tool parameter schema
    """
    name: str = ""
    description_en: str = ""
    description_zh: str = ""
    parameters: Union[List[Dict], Dict] = []

    def __init__(
            self,
            cfg: Optional[Dict] = None,
            use_zh: bool = False,
            max_retries: int = 3,
            retry_delay: int = 1,
    ):
        """Initialize the tool with configuration.
        
        Args:
            cfg: Tool configuration dict
            use_zh: Whether to use Chinese language
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.cfg = cfg or {}
        if "name" in self.cfg and len(self.cfg["name"]) > 0:
            self.tool_name = self.cfg["name"]
        else:
            self.tool_name = self.name
        if "name_for_model" in self.cfg and len(self.cfg["name_for_model"]) > 0:
            self.name_for_model = self.cfg["name_for_model"]
        else:
            self.name_for_model = self.name
        if "name_for_human" in self.cfg and len(self.cfg["name_for_human"]) > 0:
            self.name_for_human = self.cfg["name_for_human"]
        else:
            self.name_for_human = self.name

        if use_zh:
            if "description_zh" in self.cfg and len(self.cfg["description_zh"]) > 0:
                self.tool_description = self.cfg["description_zh"]
            else:
                self.tool_description = self.description_zh
        else:
            if "description_en" in self.cfg and len(self.cfg["description_en"]) > 0:
                self.tool_description = self.cfg["description_en"]
            else:
                self.tool_description = self.description_en

        if "parameters" in self.cfg and len(self.cfg["parameters"]) > 0:
            self.parameters = self.cfg["parameters"]

        self.use_zh = use_zh
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @abstractmethod
    def call(
            self,
            params: Union[str, Dict],
    ):
        """Execute the tool with given parameters. Must be implemented by subclasses.
        
        Args:
            params: Tool parameters as JSON string or dict
            
        Returns:
            Tool execution result
        """
        raise NotImplementedError

    @staticmethod
    def json_loads(text: str) -> Dict:
        """Parse JSON text, handling code block formatting.
        
        Args:
            text: JSON string (may contain markdown code blocks)
            
        Returns:
            Parsed dictionary
        """
        text = text.strip('\n')
        if text.startswith('```') and text.endswith('\n```'):
            text = '\n'.join(text.split('\n')[1:-1])
        try:
            return json.loads(text)
        except json.decoder.JSONDecodeError as json_err:
            raise json_err

    def verify_json_format_args(
            self,
            params: Union[str, Dict],
            strict_json: bool = False
    ):
        """Verify and parse parameters according to tool's parameter schema.
        
        Args:
            params: Parameters as JSON string or dict
            strict_json: Whether to use strict JSON parsing
            
        Returns:
            Validated parameter dictionary
            
        Raises:
            ValueError: If parameters are invalid
        """
        if isinstance(params, str):
            try:
                if strict_json:
                    params_json: Dict = json.loads(params)
                else:
                    params_json: Dict = self.json_loads(params)
            except json.decoder.JSONDecodeError:
                raise ValueError('Parameters must be formatted as a valid JSON!')
        else:
            params_json: Dict = params

        if isinstance(self.parameters, List):
            for param in self.parameters:
                if "required" in param and param["required"]:
                    if param["name"] not in params_json:
                        raise ValueError(f"Parameter {param['name']} is required")
        elif isinstance(self.parameters, Dict):
            jsonschema.validate(instance=params_json, schema=self.parameters)

        return params_json