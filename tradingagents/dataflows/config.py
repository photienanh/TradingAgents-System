import tradingagents.default_config as default_config
from typing import Any, Dict

# Use default config but allow it to be overridden
_config: Dict[str, Any] = default_config.DEFAULT_CONFIG.copy()


def initialize_config():
    """Initialize the configuration with default values."""
    global _config
    _config = default_config.DEFAULT_CONFIG.copy()


def set_config(config: Dict):
    """Update the configuration with custom values."""
    global _config
    _config.update(config)


def get_config() -> Dict:
    """Get the current configuration."""
    return _config.copy()


# Initialize with default config
initialize_config()
