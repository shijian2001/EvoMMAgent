"""Multimodal Agent for handling text, image, and video inputs.

This agent extends BasicAgent to work with vision-language models,
supporting multimodal inputs (images, videos, or mixed) and tool calling.
"""

import os
import logging
import json
from typing import List, Dict, Optional, Union, Any

from agent.base_agent import BasicAgent
from api import APIPool, load_api_keys

logger = logging.getLogger(__name__)


class MultimodalAgent(BasicAgent):
    """Agent that can process multimodal inputs (text, images, videos) and use tools.
    
    Supports Qwen-VL, DeepSeek-VL, Gemini-VL (TODO), and GPT-VL (TODO) models.
    Uses ReAct pattern for tool calling with multimodal context.
    """
    
    def __init__(
            self,
            name: str = "MultimodalAgent",
            description_en: str = "A multimodal agent that can process text, images, and videos",
            description_zh: str = "一个可以处理文本、图像和视频的多模态智能体",
            tool_bank: Optional[List[Union[str, Dict]]] = None,
            use_zh: bool = False,
            model_name: str = "qwen2.5-vl-72b-instruct",
            api_keys: Optional[List[str]] = None,
            max_iterations: Optional[int] = None,
            system_template_dir: str = "./template",
            tool_description_template_en_file: str = "ToolCaller_EN.jinja2",
            tool_description_template_zh_file: str = "ToolCaller_ZH.jinja2",
            mm_agent_template_en_file: str = "MMAgent_EN.jinja2",
            mm_agent_template_zh_file: str = "MMAgent_ZH.jinja2",
            max_concurrent_per_key: Optional[int] = None,
            base_url: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            enable_memory: Optional[bool] = None,
            memory_dir: Optional[str] = None,
            preload_devices: Optional[List[str]] = None,
            config: Optional[Any] = None,
    ):
        """Initialize the multimodal agent.
        
        Args:
            name: Agent name
            description_en: English description
            description_zh: Chinese description
            tool_bank: List of tools (str names or dict configs)
            use_zh: Whether to use Chinese language
            model_name: Name of the vision-language model to use
            api_keys: Optional list of API keys (auto-loaded from env if not provided)
            max_iterations: Maximum ReAct iterations (uses config default if None)
            system_template_dir: Directory for Jinja2 templates
            tool_description_template_en_file: English tool description template
            tool_description_template_zh_file: Chinese tool description template
            mm_agent_template_en_file: English multimodal agent system prompt template
            mm_agent_template_zh_file: Chinese multimodal agent system prompt template
            max_concurrent_per_key: Maximum concurrent requests per API key (uses config default if None)
            base_url: Base URL for API endpoint (uses config default if None)
            temperature: Sampling temperature (uses config default if None)
            max_tokens: Maximum tokens in response (uses config default if None)
            enable_memory: Whether to enable memory system (uses config default if None)
            memory_dir: Directory for memory storage (uses config default if None)
            preload_devices: Devices for preloading models (inherited from base_agent)
            config: Optional Config object for default values (auto-loads if None)
        """
        # Load config for defaults
        if config is None:
            from config import Config
            config = Config.default()
        self.config = config
        
        # Apply config defaults if parameters not specified
        if max_iterations is None:
            max_iterations = config.agent.max_iterations
        if temperature is None:
            temperature = config.agent.temperature
        if max_tokens is None:
            max_tokens = config.agent.max_tokens
        if enable_memory is None:
            enable_memory = config.agent.enable_memory
        if memory_dir is None:
            memory_dir = config.agent.memory_dir
        if max_concurrent_per_key is None:
            max_concurrent_per_key = config.api.max_concurrent_per_key
        if base_url is None:
            base_url = config.api.base_url
        
        super().__init__(
            name=name,
            description_en=description_en,
            description_zh=description_zh,
            tool_bank=tool_bank,
            use_zh=use_zh,
            system_template_dir=system_template_dir,
            tool_description_template_en_file=tool_description_template_en_file,
            tool_description_template_zh_file=tool_description_template_zh_file,
            preload_devices=preload_devices,
        )
        
        self.model_name = model_name.lower()
        self.max_iterations = max_iterations
        self.mm_agent_template_file = mm_agent_template_zh_file if use_zh else mm_agent_template_en_file
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_memory = enable_memory
        self.memory_dir = memory_dir
        
        # Auto-load API keys if not provided
        if api_keys is None:
            try:
                api_keys = load_api_keys()
            except Exception as e:
                raise ValueError(
                    f"Failed to load API keys: {str(e)}. "
                    "Please provide api_keys or ensure keys.env is configured correctly."
                )
        
        # Create API pool (disable JSON parsing for Agent compatibility)
        self.api_pool = APIPool(
            model_name=model_name,
            api_keys=api_keys,
            max_concurrent_per_key=max_concurrent_per_key,
            base_url=base_url,
            parse_json=False,  # Agent needs raw string to detect tool calls
        )
    
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions using Jinja2 template.
        
        Returns:
            System prompt string
        """
        if not self.tool_bank:
            # Simple prompt without tools
            if self.use_zh:
                return "你是一个智能助手，可以理解和处理文本、图像和视频。"
            else:
                return "You are an intelligent assistant that can understand and process text, images, and videos."
        
        # Get tool information
        tool_names, tool_descriptions = self._get_tool_description()
        
        # Prepare template variables
        template_vars = {
            "tool_names": tool_names,
            "tool_description": tool_descriptions,
            "special_think_token": self.special_think_token,
            "special_func_token": self.special_func_token,
            "special_args_token": self.special_args_token,
            "special_obs_token": self.special_obs_token,
        }
        
        # Render template
        template = self.jinja_env.get_template(self.mm_agent_template_file)
        system_prompt = template.render(**template_vars)
        
        return system_prompt
    
    
    async def _call_llm(
            self,
            system_prompt: str,
            user_prompt: List[Dict[str, Any]],
    ) -> str:
        """Call the LLM using the API pool.
        
        Args:
            system_prompt: System prompt string
            user_prompt: User prompt with multimodal content
            
        Returns:
            LLM response string
        """
        # Use the unified qa method from API pool
        result = await self.api_pool.execute(
            "qa",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        # Extract answer from result dict
        return result.get("answer", "")
    
    async def act(
            self,
            query: str,
            images: Optional[List[Union[str, Dict]]] = None,
            videos: Optional[List[Union[str, Dict]]] = None,
            verbose: bool = True,
            return_history: bool = False,
    ) -> Union[str, Dict]:
        """Execute a multimodal query with tool support using ReAct pattern.
        
        Args:
            query: Text query
            images: Optional list of image paths/URLs or dicts with image params
            videos: Optional list of video paths/URLs or dicts with video params
            verbose: Whether to print execution steps
            return_history: Whether to return full execution history
            
        Returns:
            Final response string, or dict with response and history if return_history=True
        """
        if verbose:
            logger.info(f"\n{'='*80}")
            logger.info(f"📝 USER QUERY")
            logger.info(f"{'='*80}")
            logger.info(f"Query: {query}")
            if images:
                logger.info(f"Images: {len(images)} image(s)")
            if videos:
                logger.info(f"Videos: {len(videos)} video(s)")
            logger.info(f"{'='*80}\n")
        
        # Initialize memory if enabled
        memory = None
        task_id = None
        if self.enable_memory:
            from mm_memory import Memory
            memory = Memory(base_dir=self.memory_dir)
            task_id = memory.start_task(query)
            
            if verbose:
                logger.info(f"💾 Memory enabled: Task ID = {task_id}\n")
            
            # Register inputs
            if images:
                for img in images:
                    img_path = img if isinstance(img, str) else img.get("image")
                    if img_path and os.path.exists(img_path):
                        memory.add_input(img_path, "img")
            
            if videos:
                for vid in videos:
                    vid_path = vid if isinstance(vid, str) else vid.get("video")
                    if vid_path and os.path.exists(vid_path):
                        memory.add_input(vid_path, "vid")
        
        # Build system prompt with tool descriptions
        system_prompt = self._build_system_prompt()
        
        # Build initial user prompt with multimodal content
        # If memory is enabled, prepend available resource IDs to query
        if memory:
            available_refs = []
            if images:
                # Get registered image IDs from memory
                for idx in range(len(memory.trace_data["input"].get("images", []))):
                    available_refs.append(memory.trace_data["input"]["images"][idx]["id"])
            
            if available_refs:
                refs_text = ", ".join(available_refs)
                query_with_refs = f"Available images: {refs_text}\n\n{query}"
            else:
                query_with_refs = query
        else:
            query_with_refs = query
        
        user_content = [{"type": "text", "text": query_with_refs}]
        
        # Add images
        if images:
            for img in images:
                if isinstance(img, str):
                    user_content.append({"type": "image", "image": img})
                elif isinstance(img, dict):
                    user_content.append({"type": "image", **img})
        
        # Add videos
        if videos:
            for vid in videos:
                if isinstance(vid, str):
                    user_content.append({"type": "video", "video": vid})
                elif isinstance(vid, dict):
                    user_content.append({"type": "video", **vid})
        
        # ReAct loop
        history = []
        conversation_context = user_content.copy()
        
        for iteration in range(self.max_iterations):
            if verbose:
                logger.info(f"\n{'─'*80}")
                logger.info(f"🔄 ITERATION {iteration + 1}")
                logger.info(f"{'─'*80}")
            
            # Call LLM
            try:
                if verbose:
                    logger.info(f"\n📥 INPUT TO LLM:")
                    # Show conversation context (without system)
                    for idx, msg in enumerate(conversation_context, 1):
                        if msg.get("type") == "text":
                            text = msg.get("text", "")
                            logger.info(f"   [{idx}] {text}")
                        else:
                            logger.info(f"   [{idx}] [multimodal: {msg.get('type')}]")
                
                response = await self._call_llm(system_prompt, conversation_context)
                
                if verbose:
                    logger.info(f"\n📤 LLM RESPONSE:")
                    logger.info(f"{response}")
                    
            except Exception as e:
                error_msg = f"Error calling LLM: {str(e)}"
                logger.error(error_msg)
                if return_history:
                    return {
                        "response": error_msg,
                        "history": history,
                        "success": False,
                    }
                return error_msg
            
            has_tool, tool_name, tool_args, thought = self._detect_tool(response)
            
            if has_tool:
                if memory and thought:
                    memory.log_think(thought, self.special_think_token)
                
                original_properties = None
                resolved_args_str = tool_args
                
                if memory:
                    try:
                        tool_args_dict = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                        if isinstance(tool_args_dict, dict):
                            original_properties = tool_args_dict.copy()
                            resolved_args = memory.resolve_ids(tool_args_dict)
                            resolved_args_str = json.dumps(resolved_args)
                    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                        logger.debug(f"Failed to resolve IDs in tool args: {e}")
                        pass
                
                tool_result = self._call_tool(tool_name, resolved_args_str)
                
                if isinstance(tool_result, dict):
                    output_image = tool_result.get("output_image")
                    output_video = tool_result.get("output_video")
                    
                    if output_image:
                        output_object = output_image
                        output_type = "img"
                    elif output_video:
                        output_object = output_video
                        output_type = "vid"
                    else:
                        output_object = None
                        output_type = None
                    
                    observation_data = {k: v for k, v in tool_result.items() 
                                       if k not in ["output_image", "output_video"]}
                    
                    if output_object:
                        observation = None
                    else:
                        if memory:
                            observation_data = memory.resolve_paths_to_ids(observation_data)
                        observation = self._format_observation(observation_data, tool_name)
                
                elif isinstance(tool_result, str):
                    # Legacy string return
                    observation = tool_result
                    output_object = None
                    output_type = None
                    observation_data = None
                else:
                    # PIL.Image or other object (legacy visualize_regions)
                    from PIL import Image
                    if isinstance(tool_result, Image.Image):
                        output_object = tool_result
                        output_type = "img"
                        observation_data = {}
                        observation = None
                    else:
                        observation = str(tool_result)
                        output_object = None
                        output_type = None
                        observation_data = None
                
                if verbose:
                    logger.info(f"\n🔧 TOOL EXECUTION: {tool_name}")
                    logger.info(f"   Input: {tool_args}")
                    logger.info(f"   Output: {observation}")
                
                # Log action to memory
                if memory:
                    try:
                        # Use original properties (with IDs) for trace
                        properties = original_properties if original_properties is not None else (
                            json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                        )
                        
                        if output_object and output_type:
                            # Generate description before logging
                            from tool.base_tool import TOOL_REGISTRY
                            tool_instance = TOOL_REGISTRY.get(tool_name)
                            if tool_instance:
                                description = tool_instance().generate_description(
                                    properties, 
                                    observation_data if isinstance(observation_data, dict) else {}
                                )
                            else:
                                description = f"{tool_name} output"
                            
                            # Log action with pre-generated description
                            output_id = memory.log_action(
                                tool=tool_name,
                                properties=properties,
                                observation=observation_data or {},
                                output_object=output_object,
                                output_type=output_type,
                                description=description
                            )
                            
                            # Format observation for LLM
                            if output_id:
                                obs_parts = [f"Saved as {output_id}: {description}"]
                                
                                # Add non-multimodal data if present (e.g., regions, similarity)
                                if observation_data:
                                    formatted_data = self._format_observation(observation_data, tool_name)
                                    obs_parts.append(formatted_data)
                                
                                observation = ". ".join(obs_parts) if len(obs_parts) > 1 else obs_parts[0]
                            else:
                                observation = f"Output Saved"
                        else:
                            # No multimodal output - observation already formatted
                            memory.log_action(
                                tool=tool_name,
                                properties=properties,
                                observation=observation
                            )
                    except Exception as e:
                        if verbose:
                            logger.warning(f"Failed to log action to memory: {e}")
                
                # Record history
                history.append({
                    "iteration": iteration + 1,
                    "thought": thought,
                    "action": tool_name,
                    "action_input": tool_args,
                    "observation": observation,
                })
                
                context_text = f"{thought}\n{self.special_func_token} {tool_name}\n{self.special_args_token} {tool_args}\n{self.special_obs_token} {observation}"
                conversation_context.append({
                    "type": "text",
                    "text": context_text
                })
                
            else:
                if memory:
                    memory.log_answer(response)
                    memory.end_task(success=True)
                    if verbose:
                        logger.info(f"💾 Saved trace to: {memory.task_dir}/trace.json")
                
                history.append({
                    "iteration": iteration + 1,
                    "final_response": response,
                })
                
                if verbose:
                    logger.info(f"\n{'='*80}")
                    logger.info("✅ TASK COMPLETED!")
                    logger.info(f"{'='*80}")
                
                result = {
                    "response": response,
                    "history": history,
                    "success": True,
                }
                if task_id:
                    result["task_id"] = task_id
                
                if return_history:
                    return result
                return response
        
        final_msg = "Maximum iterations reached without completing the task."
        logger.warning(final_msg)
        
        if memory:
            memory.end_task(success=False)
            if verbose:
                logger.info(f"💾 Saved incomplete trace to: {memory.task_dir}/trace.json")
        
        result = {
            "response": final_msg,
            "history": history,
            "success": False,
        }
        if task_id:
            result["task_id"] = task_id
        
        if return_history:
            return result
        return final_msg

