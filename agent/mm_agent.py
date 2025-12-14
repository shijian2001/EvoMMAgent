"""Multimodal Agent for handling text, image, and video inputs.

This agent extends BasicAgent to work with vision-language models,
supporting multimodal inputs (images, videos, or mixed) and tool calling.
"""

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
        
        # Build system prompt with tool descriptions
        system_prompt = self._build_system_prompt()
        
        # Build initial user prompt with multimodal content
        user_content = [{"type": "text", "text": query}]
        
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
                # Execute tool
                observation = self._call_tool(tool_name, tool_args)
                
                if verbose:
                    logging.info(f"\n🔧 TOOL EXECUTION: {tool_name}")
                    logging.info(f"   Input: {tool_args}")
                    obs_preview = str(observation)[:500] + "..." if len(str(observation)) > 500 else str(observation)
                    logging.info(f"   Output: {obs_preview}")
                
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
                history.append({
                    "iteration": iteration + 1,
                    "final_response": response,
                })
                
                if verbose:
                    logging.info(f"\n{'='*80}")
                    logging.info("✅ TASK COMPLETED!")
                    logging.info(f"{'='*80}")
                
                if return_history:
                    return {
                        "response": response,
                        "history": history,
                        "success": True,
                    }
                return response
        
        # Max iterations reached
        final_msg = "Maximum iterations reached without completing the task."
        logging.warning(final_msg)
        
        if return_history:
            return {
                "response": final_msg,
                "history": history,
                "success": False,
            }
        return final_msg

