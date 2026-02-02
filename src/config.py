"""
Configuration Module

Centralized configuration management for the infinite debate system.
"""

from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path


@dataclass
class SystemConfig:
    """Main system configuration."""
    
    # API Configuration
    groq_api_key: str
    
    # Storage Configuration
    storage_dir: str = "./conversations"
    log_dir: str = "./logs"
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Default Protocol Parameters
    default_temperature: float = 1.0
    default_max_tokens: int = 1024
    default_model: str = "llama-3.1-8b-instant"
    
    # Performance Configuration
    turn_timeout_seconds: int = 30
    max_context_window: Optional[int] = None
    
    def __post_init__(self):
        """Validate and setup configuration."""
        # Create directories if they don't exist
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup log file path if not specified
        if self.log_file is None:
            self.log_file = str(Path(self.log_dir) / "debate_system.log")


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    
    agent_id: str
    stance: Optional[str] = None
    personality: Optional[str] = None
    system_prompt_template: Optional[str] = None
    
    # Agent-specific overrides
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ConfigLoader:
    """
    Loads configuration from environment variables and files.
    """
    
    @staticmethod
    def load_from_env() -> SystemConfig:
        """
        Load system configuration from environment variables.
        
        Returns:
            SystemConfig instance
        
        Raises:
            ValueError: If required configuration is missing
        """
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is required. "
                "Copy .env.example to .env and set your API key."
            )
        
        return SystemConfig(
            groq_api_key=api_key,
            storage_dir=os.environ.get("STORAGE_DIR", "./conversations"),
            log_dir=os.environ.get("LOG_DIR", "./logs"),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            log_file=os.environ.get("LOG_FILE"),
            default_temperature=float(os.environ.get("DEFAULT_TEMPERATURE", "1.0")),
            default_max_tokens=int(os.environ.get("DEFAULT_MAX_TOKENS", "1024")),
            default_model=os.environ.get("DEFAULT_MODEL", "llama-3.1-8b-instant"),
            turn_timeout_seconds=int(os.environ.get("TURN_TIMEOUT_SECONDS", "30"))
        )
    
    @staticmethod
    def load_from_dict(config_dict: dict) -> SystemConfig:
        """
        Load system configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            SystemConfig instance
        """
        return SystemConfig(**config_dict)


# Example preset configurations for different debate scenarios

PRESET_CONFIGS = {
    "fast_debate": {
        "default_temperature": 0.7,
        "default_max_tokens": 512,
        "turn_timeout_seconds": 15
    },
    "deep_debate": {
        "default_temperature": 1.0,
        "default_max_tokens": 2048,
        "turn_timeout_seconds": 60
    },
    "creative_debate": {
        "default_temperature": 1.2,
        "default_max_tokens": 1024,
        "turn_timeout_seconds": 30
    }
}


def get_preset_config(preset_name: str, base_config: SystemConfig) -> SystemConfig:
    """
    Apply a preset configuration to a base config.
    
    Args:
        preset_name: Name of preset configuration
        base_config: Base system configuration
    
    Returns:
        Updated SystemConfig
    
    Raises:
        ValueError: If preset name is unknown
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available: {list(PRESET_CONFIGS.keys())}"
        )
    
    preset = PRESET_CONFIGS[preset_name]
    
    # Create new config with updated values
    return SystemConfig(
        groq_api_key=base_config.groq_api_key,
        storage_dir=base_config.storage_dir,
        log_dir=base_config.log_dir,
        log_level=base_config.log_level,
        log_file=base_config.log_file,
        default_temperature=preset.get("default_temperature", base_config.default_temperature),
        default_max_tokens=preset.get("default_max_tokens", base_config.default_max_tokens),
        default_model=preset.get("default_model", base_config.default_model),
        turn_timeout_seconds=preset.get("turn_timeout_seconds", base_config.turn_timeout_seconds),
        max_context_window=base_config.max_context_window
    )
