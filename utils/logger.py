"""Logger utility for consistent logging across the project."""

import os
import logging
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "EvoMMAgent",
    log_dir: str = "logs",
    level: int = logging.INFO,
    console: bool = True,
    file_logging: bool = True,
) -> logging.Logger:
    """Setup a logger that writes to both console and file.
    
    Args:
        name: Logger name (used as filename prefix)
        log_dir: Directory to store log files
        level: Logging level (default: INFO)
        console: Whether to log to console (default: True)
        file_logging: Whether to log to file (default: True)
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger("mm_agent")
        >>> logger.info("Starting test...")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_logging:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        print(f"Logging to: {log_file}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = "EvoMMAgent") -> logging.Logger:
    """Get an existing logger or create a basic one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

