import asyncio

from itertools import cycle
from typing import List


class ApiKeyCycler:
    """Thread-safe API key cycler for load balancing across multiple API keys.
    
    Cycles through a list of API keys in round-robin fashion with async lock protection.
    """
    
    def __init__(self, api_key_list: List[str]):
        """Initialize the API key cycler.
        
        Args:
            api_key_list: List of API keys to cycle through
        """
        self.api_keys = api_key_list
        self._lock = asyncio.Lock()
        self._cycler = cycle(self.api_keys)

    async def get_key(self) -> str:
        """Get the next API key in the cycle.
        
        Thread-safe method that returns the next key in round-robin order.
        
        Returns:
            Next API key string
        """
        async with self._lock:
            return next(self._cycler)
