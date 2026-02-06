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
    
    # Memory bank: points to the training memory dir (bank/ lives inside it)
    bank_memory_dir: str = ""
    
    # Query Rewriter
    enable_query_rewrite: bool = True
    query_rewrite_strategy: str = "text_only"   # "text_only" | "auto" (future)
    max_sub_queries: int = 3
    
    # Embedding Retrieval (vLLM deployed)
    embedding_model: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    retrieval_top_k: int = 10
    
    # Rerank (vLLM deployed)
    enable_rerank: bool = True
    rerank_model: str = ""
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_top_n: int = 3
    
    # Multi-round Deep Research
    max_retrieval_rounds: int = 1               # 1=single, 2+=multi-round


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
