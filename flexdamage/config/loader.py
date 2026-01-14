import yaml
from pathlib import Path
from typing import Union, Dict, Any
from .schema import RunConfig

def load_config(path: Union[str, Path]) -> RunConfig:
    """
    Load and validate a YAML configuration file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
        
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
        
    return RunConfig(**data)

def save_config(config: RunConfig, path: Union[str, Path]):
    """
    Save configuration to YAML.
    """
    with open(path, 'w') as f:
        # exclude_none=True to keep config clean
        yaml.dump(config.model_dump(exclude_none=True), f)
