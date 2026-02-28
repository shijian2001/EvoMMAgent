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
class RetrievalConfig:
    """Retrieval-augmented generation settings. enable=False skips all retrieval."""
    
    # Master switch
    enable: bool = False
    
    # Retrieval mode: "trace" (task-level) or "state" (state-level MDP)
    mode: str = "state"
    
    # Memory bank: points to the training memory dir (trace_bank/ or state_bank/ lives inside it)
    bank_memory_dir: str = ""
    bank_dir_name: str = ""                      # override bank subfolder name (default: trace_bank / state_bank)
    
    # Embedding Retrieval (vLLM deployed, shared by both modes)
    embedding_model: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    # Search quality filter (shared)
    min_score: float = 0.1                      # cosine similarity threshold
    
    # === Trace mode settings ===
    trace_top_n: int = 1                         # fixed top-1 retrieval
    
    # === State mode settings ===
    min_q_value: int = 7                        # Q-value filter threshold
    experience_top_n: int = 1                   # number of experiences to inject per step
    max_epoch: int = 1                          # max retrieval rounds per state


@dataclass
class Config:
    """Main configuration container."""
    
    agent: AgentConfig
    api: APIConfig
    retrieval: RetrievalConfig
    
    def __init__(
        self,
        agent: Optional[AgentConfig] = None,
        api: Optional[APIConfig] = None,
        retrieval: Optional[RetrievalConfig] = None,
    ):
        """Initialize config with default or provided values."""
        self.agent = agent or AgentConfig()
        self.api = api or APIConfig()
        self.retrieval = retrieval or RetrievalConfig()
    
    @classmethod
    def default(cls) -> "Config":
        """Get default configuration."""
        return cls()


# Global default config instance
default_config = Config.default()
