import logging
import sys
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO):
    """Configure logger with custom formatting and multiple handlers"""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(logs_dir / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Create separate loggers for different components