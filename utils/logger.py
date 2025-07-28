"""Logging utility functions."""
import logging
import logging.handlers
import sys
import os
from config.system_config import system_config

class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI colors based on log level."""
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
        'ENDC': '\033[0m'
    }

    def format(self, record: logging.LogRecord) -> str:
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['ENDC']}"
        return super().format(record)


def setup_logger(log_dir: str = "logs") -> None:
    """Configure root logger handlers."""
    os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, system_config.log_level.upper()))
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    colored_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(colored_formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'quant_system.log'),
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    error_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'error.log'),
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the specified name."""
    return logging.getLogger(name)


# Initialize logging on import
setup_logger()
