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
            max_retries: Optional[int] = None,
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
            max_retries: Maximum retry attempts for failed API requests (uses config default if None)
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
        if max_retries is None:
            max_retries = config.api.max_retries
        
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
            max_retries=max_retries,
            parse_json=False,  # Agent needs raw string to detect tool calls
        )
    
    
    def _build_system_prompt(self) -> str:
        """Build simplified system prompt using Jinja2 template.
        
        Tool definitions are passed separately via tools parameter to API,
        not included in the system prompt.
        
        Returns:
            System prompt string
        """
        if not self.tool_bank:
            # Simple prompt without tools (for direct query)
            base_prompt = (
                "你是一个可以理解文本、图像和视频的多模态助手。请仔细分析问题并给出清晰的答案。"
            ) if self.use_zh else (
                "You are a multimodal assistant that can understand text, images, and videos. "
                "Analyze the question carefully and provide clear answers."
            )
            return base_prompt
        
        # Use template for tool-enabled agent
        template = self.jinja_env.get_template(self.mm_agent_template_file)
        system_prompt = template.render(enable_memory=self.enable_memory)
        
        return system_prompt
    
    
    async def _call_llm(
            self,
            system_prompt: str,
            conversation_history: List[Dict[str, Any]],
            tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Call the LLM using the API pool.
        
        Args:
            system_prompt: System prompt string
            conversation_history: List of messages in OpenAI format with roles (user/assistant/tool)
            tools: Optional list of tool definitions
            
        Returns:
            Dict with answer, tool_calls (if any), and other metadata
        """
        # Use the unified qa method from API pool with new signature
        result = await self.api_pool.execute(
            "qa",
            system=system_prompt,
            messages=conversation_history,
            tools=tools,
            tool_choice="auto" if tools else "none",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        return result
    
    async def act(
            self,
            query: str,
            images: Optional[List[Union[str, Dict]]] = None,
            videos: Optional[List[Union[str, Dict]]] = None,
            verbose: bool = True,
            return_history: bool = False,
            task_metadata: Optional[Dict] = None,
    ) -> Union[str, Dict]:
        """Execute a multimodal query with tool support using ReAct pattern.
        
        Args:
            query: Text query
            images: Optional list of image paths/URLs or dicts with image params
            videos: Optional list of video paths/URLs or dicts with video params
            verbose: Whether to print execution steps
            return_history: Whether to return full execution history
            task_metadata: Optional metadata for memory trace (dataset_id, dataset, sub_task, type, etc.)
            
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
            
            # Extract dataset_id and metadata from task_metadata
            dataset_id = None
            metadata_kwargs = {}
            if task_metadata:
                dataset_id = task_metadata.get("dataset_id")
                # Pass all other metadata keys
                for key, value in task_metadata.items():
                    if key != "dataset_id":
                        metadata_kwargs[key] = value
            
            task_id = memory.start_task(query, dataset_id=dataset_id, **metadata_kwargs)
            
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
        
        # Build system prompt (simplified, without tool descriptions)
        system_prompt = self._build_system_prompt()
        
        # Build tools schema for API
        tools_schema = self._build_tools_schema() if self.tool_bank else None
        
        # Build initial user message with multimodal content
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
        
        # Build user message content (multimodal content list)
        initial_user_content = [{"type": "text", "text": query_with_refs}]
        
        # Add images (marked as initial for memory system)
        if images:
            for img in images:
                if isinstance(img, str):
                    initial_user_content.append({"type": "image", "image": img, "initial": True})
                elif isinstance(img, dict):
                    initial_user_content.append({"type": "image", "initial": True, **img})
        
        # Add videos (marked as initial for memory system)
        if videos:
            for vid in videos:
                if isinstance(vid, str):
                    initial_user_content.append({"type": "video", "video": vid, "initial": True})
                elif isinstance(vid, dict):
                    initial_user_content.append({"type": "video", "initial": True, **vid})
        
        # Initialize conversation history in OpenAI message format
        # conversation_history will accumulate: user, assistant, tool messages
        conversation_history = [
            {"role": "user", "content": initial_user_content}
        ]
        
        # Track execution history for logging
        history = []
        
        for iteration in range(self.max_iterations):
            # Remove input images from first user message starting from second iteration
            # Also update text prompt to reflect images are now in memory
            if iteration == 1 and conversation_history:
                first_msg = conversation_history[0]
                if first_msg.get("role") == "user" and isinstance(first_msg.get("content"), list):
                    new_content = []
                    for item in first_msg["content"]:
                        if not item.get("initial"):  # Remove images/videos with "initial" flag
                            if item.get("type") == "text":
                                # Update prompt text
                                text = item.get("text", "")
                                if "Available images:" in text:
                                    text = text.replace("Available images:", "Images in memory:")
                                new_content.append({"type": "text", "text": text})
                            else:
                                new_content.append(item)
                    
                    if new_content:
                        first_msg["content"] = new_content
            
            if verbose:
                logger.info(f"\n{'─'*80}")
                logger.info(f"🔄 ITERATION {iteration + 1}")
                logger.info(f"{'─'*80}")
            
            # Call LLM with full conversation history
            try:
                if verbose:
                    logger.info(f"\n📥 INPUT TO LLM:")
                    logger.info(f"   Conversation history: {len(conversation_history)} message(s)")
                    for idx, msg in enumerate(conversation_history, 1):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            logger.info(f"   [{idx}] role={role}, content items: {len(content)}")
                        elif isinstance(content, str):
                            logger.info(f"   [{idx}] role={role}, text: {content[:80]}...")
                        else:
                            logger.info(f"   [{idx}] role={role}")
                
                response = await self._call_llm(system_prompt, conversation_history, tools_schema)
                
                if verbose:
                    logger.info(f"\n📤 LLM RESPONSE:")
                    logger.info(f"Answer: {response.get('answer', '')[:200]}...")
                    if response.get('tool_calls'):
                        logger.info(f"Tool calls: {len(response['tool_calls'])} tool(s)")
                    
            except Exception as e:
                error_msg = f"Error calling LLM: {str(e)}"
                logger.error(error_msg)
                import traceback
                traceback.print_exc()
                if return_history:
                    return {
                        "response": error_msg,
                        "history": history,
                        "success": False,
                    }
                return error_msg
            
            # Check if model wants to call tools
            tool_calls = response.get("tool_calls", [])
            
            if tool_calls:
                # Model wants to call tools
                assistant_content = response.get("answer", "") or ""
                
                # Log thinking if present
                if memory and assistant_content:
                    memory.log_think(assistant_content)
                
                # Add assistant message with tool_calls to conversation history
                assistant_message = {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls
                }
                conversation_history.append(assistant_message)
                
                # Process each tool call
                for tool_call in tool_calls:
                    tool_id = tool_call["id"]
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]
                    
                    # Resolve IDs if memory enabled
                original_properties = None
                resolved_args_str = tool_args
                
                if memory:
                    try:
                        tool_args_dict = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                        if isinstance(tool_args_dict, dict):
                            original_properties = tool_args_dict.copy()
                            # Special handling: get_images needs IDs, not paths
                            if tool_name != "get_images":
                                resolved_args = memory.resolve_ids(tool_args_dict)
                                resolved_args_str = json.dumps(resolved_args)
                    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                        logger.debug(f"Failed to resolve IDs in tool args: {e}")
                        pass
                
                tool_result = self._call_tool(tool_name, resolved_args_str)
                
                # Special handling for get_images tool
                if isinstance(tool_result, dict) and "_get_images_ids" in tool_result:
                    image_ids = tool_result["_get_images_ids"]
                    observation = tool_result.get("message", "Images loaded")
                    
                    # Add tool response message
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": observation
                    })
                    
                    # Remove previous get_images user messages to avoid exceeding image limit
                    # Keep only: initial user message, assistant messages, and tool messages
                    filtered_history = []
                    for i, msg in enumerate(conversation_history):
                        if msg.get("role") == "user" and i > 0:  # Not the first user message
                            # Skip user messages that contain images/videos (from previous get_images)
                            content = msg.get("content", [])
                            if isinstance(content, list):
                                has_media = any(item.get("type") in ["image", "video"] for item in content)
                                if has_media:
                                    continue  # Skip this message - it's from a previous get_images
                        filtered_history.append(msg)
                    conversation_history = filtered_history
                    
                    # Build new user message with requested images
                    new_user_content = []
                    
                    if memory:
                        for img_id in image_ids:
                            # Get description from memory
                            desc = memory.get_description(img_id) or "Generated content"
                            
                            # Add text label
                            new_user_content.append({
                                "type": "text",
                                "text": f"{img_id}: {desc}"
                            })
                            
                            # Add actual media
                            file_path = memory.get_file_path(img_id)
                            if file_path:
                                media_type = "video" if img_id.startswith("vid_") else "image"
                                new_user_content.append({
                                    "type": media_type,
                                    media_type: file_path
                                })
                    
                    # Add new user message with images to conversation
                    if new_user_content:
                        conversation_history.append({
                            "role": "user",
                            "content": new_user_content
                        })
                    
                    # Log to memory
                    if memory:
                        memory.log_action(
                            tool=tool_name,
                            properties=original_properties or (json.loads(tool_args) if isinstance(tool_args, str) else tool_args),
                            observation=observation
                        )
                    
                    # Record history
                    history.append({
                        "iteration": iteration + 1,
                        "action": tool_name,
                        "action_input": tool_args,
                        "observation": observation,
                    })
                    
                    continue  # Skip normal processing for get_images
                
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
                
                # Handle output_object (images/videos produced by tools)
                if output_object and output_type:
                    if memory:
                        # With memory: save to file and generate ID
                        try:
                            properties = original_properties if original_properties is not None else (
                                json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                            )
                            
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
                            
                            # Format observation for LLM (with ID reference)
                            if output_id:
                                obs_parts = [f"Saved to memory as {output_id}: {description}"]
                                
                                # Add non-multimodal data if present (e.g., regions, similarity)
                                if observation_data:
                                    formatted_data = self._format_observation(observation_data, tool_name)
                                    obs_parts.append(formatted_data)
                                
                                observation = ". ".join(obs_parts) if len(obs_parts) > 1 else obs_parts[0]
                            else:
                                observation = f"Output Saved"
                        except Exception as e:
                            if verbose:
                                logger.warning(f"Failed to log action to memory: {e}")
                            observation = self._format_observation(observation_data, tool_name) if observation_data else "Output generated"
                    else:
                        # Without memory: don't save, only format observation data
                        observation = self._format_observation(observation_data, tool_name) if observation_data else "Output generated"
                    
                    # Add tool message to conversation history
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": observation
                    })
                    
                    # If memory is disabled, add image/video directly to next user message
                    if not memory:
                        from PIL import Image
                        import tempfile
                        
                        user_content_with_media = []
                        
                        if output_type == "img":
                            # Save image to temp file for API compatibility
                            if isinstance(output_object, Image.Image):
                                os.makedirs("temp_images", exist_ok=True)
                                temp_file = tempfile.NamedTemporaryFile(
                                    delete=False,
                                    suffix=".png",
                                    dir="temp_images"
                                )
                                output_object.save(temp_file.name)
                                image_path = temp_file.name
                            else:
                                image_path = output_object  # Already a path
                            
                            user_content_with_media.append({
                                "type": "image",
                                "image": image_path
                            })
                        elif output_type == "vid":
                            # Ensure video is a path string
                            video_path = output_object if isinstance(output_object, str) else str(output_object)
                            
                            user_content_with_media.append({
                                "type": "video",
                                "video": video_path
                            })
                        
                        # Add user message with media
                        if user_content_with_media:
                            conversation_history.append({
                                "role": "user",
                                "content": user_content_with_media
                            })
                    
                elif memory:
                    # No output_object, but memory enabled: log text-only action
                    try:
                        properties = original_properties if original_properties is not None else (
                            json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                        )
                        memory.log_action(
                            tool=tool_name,
                            properties=properties,
                            observation=observation
                        )
                    except Exception as e:
                        if verbose:
                            logger.warning(f"Failed to log action to memory: {e}")
                    
                    # Add tool message to conversation history
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": observation
                    })
                else:
                    # No output_object and no memory: add tool message
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": observation
                    })
                
                if verbose:
                    logger.info(f"\n🔧 TOOL EXECUTION: {tool_name}")
                    logger.info(f"   Input: {tool_args}")
                    logger.info(f"   Output: {observation}")
                
                # Record history
                history.append({
                    "iteration": iteration + 1,
                    "action": tool_name,
                    "action_input": tool_args,
                    "observation": observation,
                })
                
            else:
                # No tool calls - this is the final answer
                final_answer = response.get("answer", "")
                
                if memory:
                    memory.log_answer(final_answer)
                    memory.end_task(success=True)
                    if verbose:
                        logger.info(f"💾 Saved trace to: {memory.task_dir}/trace.json")
                
                # Cleanup temp images if memory was disabled
                if not memory:
                    import shutil
                    if os.path.exists("temp_images"):
                        shutil.rmtree("temp_images")
                
                history.append({
                    "iteration": iteration + 1,
                    "final_response": final_answer,
                })
                
                if verbose:
                    logger.info(f"\n{'='*80}")
                    logger.info("✅ TASK COMPLETED!")
                    logger.info(f"{'='*80}")
                
                result = {
                    "response": final_answer,
                    "history": history,
                    "success": True,
                }
                if task_id:
                    result["task_id"] = task_id
                
                if return_history:
                    return result
                return final_answer
        
        final_msg = "Maximum iterations reached without completing the task."
        logger.warning(final_msg)
        
        if memory:
            memory.end_task(success=False)
            if verbose:
                logger.info(f"💾 Saved incomplete trace to: {memory.task_dir}/trace.json")
        
        # Cleanup temp images if memory was disabled
        if not memory:
            import shutil
            if os.path.exists("temp_images"):
                shutil.rmtree("temp_images")
        
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

