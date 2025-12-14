"""API key loader from environment file."""

import os
import json
from typing import List
from dotenv import load_dotenv


def load_api_keys(env_path: str = "./api/utils/keys.env", key_name: str = "DIRECTLLM_API_KEY_USER") -> List[str]:
    """
    Load API keys from environment file.
    
    Args:
        env_path: Path to .env file (default: ./api/utils/keys.env)
        key_name: Name of the environment variable (default: DIRECTLLM_API_KEY_USER)
    
    Returns:
        List of API key strings
    
    Raises:
        ValueError: If no API keys are found or file doesn't exist
    """
    if not os.path.exists(env_path):
        raise ValueError(f"Environment file not found: {env_path}")
    
    load_dotenv(env_path)
    
    key_value = os.environ.get(key_name, "{}")
    
    try:
        key_dict = json.loads(key_value)
        if not isinstance(key_dict, dict):
            raise ValueError(f"{key_name} must be a JSON object")
        
        if not key_dict:
            raise ValueError(f"No API keys found in {key_name}")
        
        return list(key_dict.values())
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse {key_name} as JSON: {str(e)}")
