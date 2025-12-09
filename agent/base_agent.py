import logging
import traceback
import json
import jinja2

from typing import Optional, List, Union, Dict
from abc import ABC, abstractmethod
from tool.base_tool import BasicTool, ModelBasedTool, TOOL_REGISTRY
from tool.gpu_manager import acquire_gpu, release_gpu

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
            special_func_token: str = "\nAction:",
            special_args_token: str = "\nAction Input:",
            special_obs_token: str = "\nObservation:",
    ):
        """Initialize the agent with tools and configuration.
        
        Args:
            name: Agent name
            description_en: English description
            description_zh: Chinese description
            tool_bank: List of tools (str names or dict configs)
            use_zh: Whether to use Chinese language
            system_template_dir: Directory for Jinja2 templates
            tool_description_template_en_file: English tool description template
            tool_description_template_zh_file: Chinese tool description template
            special_func_token: Token to mark action/function calls
            special_args_token: Token to mark action input
            special_obs_token: Token to mark observations
        """
        self.name = name
        self.description = description_en if not use_zh else description_zh

        self.tool_bank = {}
        if tool_bank:
            for t in tool_bank:
                self._init_tool(t)
        self.use_zh = use_zh

        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(system_template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.tool_description_jinja_file = tool_description_template_zh_file if use_zh else tool_description_template_en_file
        self.special_func_token = special_func_token
        self.special_args_token = special_args_token
        self.special_obs_token = special_obs_token

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
            acquire_gpu(instance.device)
            logger.info(f'{tool_name} loaded on {instance.device}')

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

        if isinstance(tool_result, str):
            return tool_result
        else:
            return json.dumps(tool_result, ensure_ascii=False, indent=4)

    def _detect_tool(
            self,
            response: str,
    ):
        """Detect tool call from LLM response using special tokens.
        
        Args:
            response: LLM response text
            
        Returns:
            Tuple of (has_tool_call, func_name, func_args, thought)
        """
        # Reference: https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/agents/react_chat.py
        func_name, func_args = None, None
        func_idx, args_idx, obs_idx = response.rfind(self.special_func_token), response.rfind(
            self.special_args_token), response.rfind(self.special_obs_token)
        if 0 <= func_idx < args_idx:  # If the text has `Action` and `Action input`,
            if obs_idx < args_idx:  # but does not contain `Observation`,
                # then it is likely that `Observation` is ignored by the LLM,
                # because the output text may have discarded the stop word.
                response = response.rstrip() + self.special_obs_token  # Add it back.

            obs_idx = response.rfind(self.special_obs_token)
            func_name = response[func_idx + len(self.special_func_token):args_idx].strip()
            func_args = response[args_idx + len(self.special_args_token):obs_idx].strip()
            response = response[:func_idx]  # Return the response before tool call, i.e., `Thought`

        return (func_name is not None), func_name, func_args, response

    def _get_tool_description(
            self
    ):
        """Generate tool descriptions from registered tools using templates.
        
        Returns:
            Tuple of (comma-separated tool names, formatted tool descriptions)
        """
        tool_names, tool_descriptions = [], []
        for tool in self.tool_bank.values():
            tool_name = tool.tool_name
            tool_names.append(tool_name)
            name_for_model = tool.name_for_model
            name_for_human = tool.name_for_human
            tool_description = tool.tool_description
            assert len(tool_name) > 0 and len(name_for_model) > 0 and len(name_for_human) > 0 and len(
                tool_description) > 0, f"tool_name is {tool_name}"

            template_vars = {
                "name_for_model": name_for_model,
                "name_for_human": name_for_human,
                "tool_description": tool_description,
                "parameters": tool.parameters,
                "example": tool.tool_example,
            }
            jinja_file = self.tool_description_jinja_file
            template = self.jinja_env.get_template(jinja_file)
            prompt = template.render(**template_vars)
            tool_descriptions.append(prompt)

        tool_name = ",".join(tool_names)
        tool_description = "\n\n".join(tool_descriptions)
        return (tool_name, tool_description)

    def cleanup(self) -> None:
        """Cleanup all model-based tools and release GPU resources."""
        for name, t in self.tool_bank.items():
            if isinstance(t, ModelBasedTool) and t.is_loaded:
                logger.info(f'Unloading {name} from {t.device}')
                release_gpu(t.device)
                t.unload_model()
    
    def __del__(self):
        """Auto cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass
    
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