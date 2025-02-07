import yaml
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    """Loads the configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_logging(log_level="INFO"):
    """Sets up basic logging."""
    import logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )