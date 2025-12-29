"""Configuration management for EvoMMAgent."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentConfig:
    """Agent configuration settings."""
    
    # ReAct loop settings
    max_iterations: int = 10
    
    # LLM settings
    temperature: float = 0.2
    max_tokens: Optional[int] = None  # None means no limit
    
    # Memory settings
    enable_memory: bool = True
    memory_dir: str = "memory"


@dataclass
class APIConfig:
    """API configuration settings."""
    
    # API endpoint
    base_url: str = "http://redservingapi.devops.xiaohongshu.com/v1"
    
    # Concurrency settings
    max_concurrent_per_key: int = 10
    
    # Retry settings
    max_retries: int = 5
    
    # Timeout settings
    timeout: int = 120


@dataclass
class Config:
    """Main configuration container."""
    
    agent: AgentConfig
    api: APIConfig
    
    def __init__(
        self,
        agent: Optional[AgentConfig] = None,
        api: Optional[APIConfig] = None,
    ):
        """Initialize config with default or provided values."""
        self.agent = agent or AgentConfig()
        self.api = api or APIConfig()
    
    @classmethod
    def default(cls) -> "Config":
        """Get default configuration."""
        return cls()


# Global default config instance
default_config = Config.default()
