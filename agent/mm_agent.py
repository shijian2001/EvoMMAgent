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
            max_iterations: int = 10,
            system_template_dir: str = "./template",
            tool_description_template_en_file: str = "ToolCaller_EN.jinja2",
            tool_description_template_zh_file: str = "ToolCaller_ZH.jinja2",
            mm_agent_template_en_file: str = "MMAgent_EN.jinja2",
            mm_agent_template_zh_file: str = "MMAgent_ZH.jinja2",
            max_concurrent_per_key: int = 10,
            base_url: str = "http://redservingapi.devops.xiaohongshu.com/v1",
            temperature: float = 1.0,
            max_tokens: Optional[int] = None,
            enable_memory: bool = True,
            memory_dir: str = "memory",
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
            max_iterations: Maximum ReAct iterations
            system_template_dir: Directory for Jinja2 templates
            tool_description_template_en_file: English tool description template
            tool_description_template_zh_file: Chinese tool description template
            mm_agent_template_en_file: English multimodal agent system prompt template
            mm_agent_template_zh_file: Chinese multimodal agent system prompt template
            max_concurrent_per_key: Maximum concurrent requests per API key
            base_url: Base URL for API endpoint
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            enable_memory: Whether to enable memory system for saving traces
            memory_dir: Directory for memory storage
        """
        super().__init__(
            name=name,
            description_en=description_en,
            description_zh=description_zh,
            tool_bank=tool_bank,
            use_zh=use_zh,
            system_template_dir=system_template_dir,
            tool_description_template_en_file=tool_description_template_en_file,
            tool_description_template_zh_file=tool_description_template_zh_file,
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
            logging.info(f"\n{'='*80}")
            logging.info(f"📝 USER QUERY")
            logging.info(f"{'='*80}")
            logging.info(f"Query: {query}")
            if images:
                logging.info(f"Images: {len(images)} image(s)")
            if videos:
                logging.info(f"Videos: {len(videos)} video(s)")
            logging.info(f"{'='*80}\n")
        
        # Initialize memory if enabled
        memory = None
        task_id = None
        if self.enable_memory:
            from mm_memory import Memory
            memory = Memory(base_dir=self.memory_dir)
            task_id = memory.start_task(query)
            
            if verbose:
                logging.info(f"💾 Memory enabled: Task ID = {task_id}\n")
            
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
                logging.info(f"\n{'─'*80}")
                logging.info(f"🔄 ITERATION {iteration + 1}")
                logging.info(f"{'─'*80}")
            
            # Call LLM
            try:
                if verbose:
                    logging.info(f"\n📥 INPUT TO LLM:")
                    # Show conversation context (without system)
                    for idx, msg in enumerate(conversation_context, 1):
                        if msg.get("type") == "text":
                            text = msg.get("text", "")
                            preview = text[:300] + "..." if len(text) > 300 else text
                            logging.info(f"   [{idx}] {preview}")
                        else:
                            logging.info(f"   [{idx}] [multimodal: {msg.get('type')}]")
                
                response = await self._call_llm(system_prompt, conversation_context)
                
                if verbose:
                    logging.info(f"\n📤 LLM RESPONSE:")
                    logging.info(f"{response}")
                    
            except Exception as e:
                error_msg = f"Error calling LLM: {str(e)}"
                logging.error(error_msg)
                if return_history:
                    return {
                        "response": error_msg,
                        "history": history,
                        "success": False,
                    }
                return error_msg
            
            # Detect tool call
            has_tool, tool_name, tool_args, thought = self._detect_tool(response)
            
            if has_tool:
                # Log thinking to memory
                if memory and thought:
                    memory.log_think(thought, self.special_think_token)
                
                # Parse original properties (for trace)
                original_properties = None
                resolved_args_str = tool_args
                
                # Resolve IDs to file paths before calling tool
                if memory:
                    try:
                        tool_args_dict = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                        original_properties = tool_args_dict.copy()  # Save original for trace
                        resolved_args = memory.resolve_ids(tool_args_dict)
                        resolved_args_str = json.dumps(resolved_args)
                    except:
                        pass  # If parsing fails, use original args
                
                # Execute tool with resolved paths
                tool_result = self._call_tool(tool_name, resolved_args_str)
                
                # Handle different return types
                if isinstance(tool_result, dict):
                    # Tool returned dict (may contain PIL.Image)
                    output_object = tool_result.get("output_image") or tool_result.get("output_path")
                    observation_data = {k: v for k, v in tool_result.items() 
                                      if k not in ["output_image", "output_path", "success"]}
                    # Serialize for display (without PIL.Image)
                    observation = json.dumps(observation_data) if observation_data else "success"
                elif isinstance(tool_result, str):
                    # Tool returned string (legacy format or no output)
                    observation = tool_result
                    try:
                        observation_dict = json.loads(observation)
                        output_object = observation_dict.get("output_image") or observation_dict.get("output_path")
                        observation_data = {k: v for k, v in observation_dict.items() 
                                          if k not in ["output_image", "output_path", "success"]}
                    except:
                        output_object = None
                        observation_data = None
                else:
                    observation = str(tool_result)
                    output_object = None
                    observation_data = None
                
                if verbose:
                    logging.info(f"\n🔧 TOOL EXECUTION: {tool_name}")
                    logging.info(f"   Input: {tool_args}")
                    obs_preview = str(observation)[:500] + "..." if len(str(observation)) > 500 else str(observation)
                    logging.info(f"   Output: {obs_preview}")
                
                # Log action to memory
                if memory:
                    try:
                        # Use original properties (with IDs) for trace
                        properties = original_properties if original_properties is not None else (
                            json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                        )
                        
                        if output_object:
                            # Has multimodal output - let memory handle it
                            output_id = memory.log_action(
                                tool=tool_name,
                                properties=properties,
                                observation=observation_data or {},
                                output_object=output_object,  # PIL.Image or file path
                                output_type="img"
                            )
                            # Update observation with id for LLM context
                            if output_id:
                                observation = f"Output saved as {output_id}"
                        else:
                            # No multimodal output - observation as is
                            memory.log_action(
                                tool=tool_name,
                                properties=properties,
                                observation=observation
                            )
                    except Exception as e:
                        if verbose:
                            logging.warning(f"Failed to log action to memory: {e}")
                
                # Record history
                history.append({
                    "iteration": iteration + 1,
                    "thought": thought,
                    "action": tool_name,
                    "action_input": tool_args,
                    "observation": observation,
                })
                
                # Update conversation context with observation
                # Add the thought, action, and observation to context
                context_text = f"{thought}\n{self.special_func_token} {tool_name}\n{self.special_args_token} {tool_args}\n{self.special_obs_token} {observation}"
                conversation_context.append({
                    "type": "text",
                    "text": context_text
                })
                
            else:
                # No tool call, agent finished
                if memory:
                    memory.log_answer(response)
                    memory.end_task(success=True)
                    if verbose:
                        logging.info(f"💾 Saved trace to: {memory.task_dir}/trace.json")
                
                history.append({
                    "iteration": iteration + 1,
                    "final_response": response,
                })
                
                if verbose:
                    logging.info(f"\n{'='*80}")
                    logging.info("✅ TASK COMPLETED!")
                    logging.info(f"{'='*80}")
                
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
        
        # Max iterations reached
        final_msg = "Maximum iterations reached without completing the task."
        logging.warning(final_msg)
        
        if memory:
            memory.end_task(success=False)
            if verbose:
                logging.info(f"💾 Saved incomplete trace to: {memory.task_dir}/trace.json")
        
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

